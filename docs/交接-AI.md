# ResiHP 交接（AI 版）

面向接手代码的模型：结构、不变量、每个模块的契约、以及改动时最容易踩的坑。
背景与「为什么做这个项目」见 [交接-人类.md](交接-人类.md)。进度只记在 [PROGRESS.md](PROGRESS.md)。

---

## 0. 任务定义

单个 rank 永久 fail-stop（真 `SIGKILL`）后，在**幸存 rank 上原地**重规划 TP/PP/DP 三维并行、
重建通信组、迁移 `param / exp_avg / exp_avg_sq / step`，从原进度继续训练；可反复注入故障直到资源耗尽。

两条根本原则（所有验收都从这两条派生）：

**A 数值验收两段式**
- 恢复**前**：完整逻辑参数 + AdamW 状态与故障前 checkpoint 逐张量 `torch.equal`。
- 恢复**后**：与「同一 checkpoint 起点 + 新拓扑 + 新配置实际 batch + 同种子」重建的参考运行逐步一致。
- 基准是**新配置从同起点**，**不是**无故障从头跑。踢掉一个 rank 后 batch 划分与归约结构都变了，
  与无故障参考必然分岔——那是语义不同，不是精度误差。写新测试时不要拿无故障基线做锚。

**B 纯 fail-stop 确定性重路由**
- DP 重路由是纯函数 `(step, failure_signature, active_topology) → assignment`，同输入必同输出。
- 不读运行时延迟、不看进度、不做模拟到达时间。`failure_signature` = 到当前事件为止**累计**失效的 rank 集合。
- 所有存活设备一律视为健康，pᵢ≡1。`Detector` / `p_i` / `speed` / `standby` / `Algorithm 1` 这些构造
  有**全仓扫描门禁**要求命中数为 0（`tests/test_acceptance.py`）。不要为了「更真实」引入它们。

---

## 1. 代码地图

```
resihp/
├── config.py      TrainConfig + 故障表校验；字段是封闭集合，无默认值兜底
├── validate.py    共用的小校验器（non_negative_int / positive_int / unique_ranks）
├── memory.py      解析显存模型（全项目唯一计算器，125 行）
├── model.py       decoder-only Transformer；稳定全局 layer id 与逻辑参数名
├── reference.py   单进程确定性参考训练 + adamw / next_token_loss（数值锚点）
├── checkpoint.py  原子 checkpoint：tmp 写 → 完整读回校验 → os.replace
├── plan.py        ExecutionPlan / StagePlan / StateRoute / build_plan / assert_invariants
│                  / boundary_hops（583 行，全项目真值来源）
├── planner/
│   ├── tp.py      choose_tp：候选 degree 生成与确定性成员选择
│   ├── pp.py      repartition_pp / balanced_layers / owner_ranges / pipeline_phases / peak_in_flight
│   └── dp.py      DPStage / DPTopology / DPPlacement / DPAssignment / assign
├── parallel/
│   ├── tp.py      TensorParallelStage：真分片前反向（Megatron f/g 两个集合）
│   ├── pp.py      PipelineRuntime：全项目唯一运行时，1F1B + 边界 scatter/gather（451 行）
│   ├── dp.py      executor_route / dp_combine_gradients / ActivationLog
│   └── reshard.py reconstruct_full（纯函数）/ reshard_tp_state（分布式驱动）/ shard_dims
├── membership.py  free_port / connect / boundary / Supervisor（托管 TCPStore、收尸、公布成员表）
├── control.py     ControlPlane：observe / form_world / build_training_groups / safe_point /
│                  agree / torch_rank / commit_checkpoint（485 行）
├── recovery.py    PlannedRun / initial_run / recover / commit_checkpoint（纯编排 state_routes）
├── verify.py      原则 A 两个契约的唯一定义（checkpoint 比对 + 参考比对）
├── train.py       worker：单 rank 训练循环，含 SIGKILL 自杀注入
└── launch.py      唯一入口：托管 store、起 worker、不联坐、汇总退出码
```

**分层纪律**：`planner/` 是纯函数（不碰通信，可单进程穷举测试），`parallel/` 才碰通信。
通信 rank / TP shard / PP owner **只从当前 ExecutionPlan 读**，禁止从旧布局或固定拓扑隐式推导。

**rank 布局约定**：`rank = ((replica * PP + stage) * TP + tp_local)`，即每个 `(replica, stage)`
拥有连续的 TP 个 rank。改这条会连带影响 plan.py 全部索引推导。

---

## 2. 硬不变量（改代码前成立，改完仍须成立）

1. `ExecutionPlan` 是 `(config, step, version, 累计失效 rank 集合, previous plan)` 的确定性函数。
   `previous` 只用来提供**旧布局**，使迁移描述为 `previous → current` 而不是 `初始配置 → current`。
   各 rank 的 `plan.digest()` 必须完全一致，不一致即 `plan_disagreement` 停机。
2. 持久恢复状态 = `RECOVERED_STATES = ("param","exp_avg","exp_avg_sq","step")` + `iteration`（数据游标）+ RNG。
   **`grad` 刻意不在其中**：安全点只在「当前迭代完成 + AdamW step 完成」后出现，且**不存在跨安全点的
   梯度累积**，下一轮必先 `zero_grad` 重算。安全点时刻的 `param.grad` 是已被消费的旧值，搬它无语义价值。
   任何 checkpoint、任何 `StateRoute` 都不含 grad。若将来引入跨安全点梯度累积，**必须先改这条契约**。
3. checkpoint 全项目只有**一个**文件，且**每轮都写**。理由：死进程事后无法贡献任何分片，唯一还能持有
   它那份状态的就是它**活着时**写下的那个；原则 A 前半句才有意义。原子性：tmp → 读回校验 → `os.replace`，
   任何失败删 tmp、保留上一份有效 checkpoint。
4. **两种 rank 号**：plan rank = 启动时的固定身份（planner / checkpoint / ExecutionPlan 只讲这一种）；
   torch rank = 当前成员表下标，每次重建都变。翻译只在 `ControlPlane.torch_rank` 一处发生，
   **每个交给 `new_group` 的 `ranks=` 列表都必须过它**。`PipelineRuntime` 的 P2P 对端走 hop 组内下标换算
   （`_peer_rank`）。
5. **唯一性纪律**：`PipelineRuntime` 是唯一运行时（控制面驱动它，测试也直接测它，不存在测试专用调度器）；
   `memory.py` 是唯一显存公式；`recovery.recover` 是唯一恢复链路；`verify.py` 是唯一验收契约；
   `parallel/tp.TensorParallelStage` 是唯一 stage 类（TP1 就是它在单 rank 组上）；
   边界实现只有 `PipelineRuntime` 里那一套 scatter/gather。**不要新增第二套。**
6. `memory.py` 的 `in_flight_micro_batches` **没有默认值**——默认 1 会静默低估 warmup 阶段的激活峰值，
   调用方必须显式从 `planner.pp.peak_in_flight` 取。`peak_in_flight` 是**回放 1F1B 调度 +1/−1** 得出的，
   不是闭式常量：调度改了，显存预算自动跟着改。
7. 1F1B 调度定义在 `planner/pp.pipeline_phases`，**不在运行时里**。三个调用方必须对它一致，
   而其中只有 `PipelineRuntime` 能 import torch：`plan.py` 与 `planner/dp.py` 靠它算峰值激活显存。
8. `resihp/model.py` 模块级钉死 `torch.set_float32_matmul_precision("highest")`：TF32 是进程级默认且
   各 torch 版本变过，计划要求全程 FP32，这条数值契约必须显式声明而不是继承默认。
9. 项目任何地方**不读 GPU 实际显存**，只读配置。`memory_budget_bytes = null` 就是不设人为上限。
10. `VOCAB_SIZE = 256` / `SEQUENCE_LENGTH = 16` 是**运行常量**写在 `train.py`，不是配置字段
    （schema 只认计划列出的字段）。两者都要传给 planner：vocab 决定哪些 TP degree 能整除，
    seq len 决定激活预算。

---

## 3. 安全点与控制面（`control.py`）

### 九步

```
1 完成当前迭代
2 原子保存 checkpoint（每轮都做）
3 迭代边界会合，读回本轮成员表（ControlPlane.observe）
4 标记该 rank 永久失效
5 TP → PP → DP 重规划（build_plan）
6 释放旧训练组与旧 world → 在幸存者上重建 world → 建新训练组
7 恢复 / 迁移 / 重切状态（recover，执行 plan.state_routes）
8 校验计划 digest 与状态 digest 一致（agree）
9 从下一迭代继续
```

### 三类通信实体

| 实体 | 后端 | 生命周期与作用 |
|---|---|---|
| world 组 | Gloo | 每次 fail-stop 重建一次。含死进程的那个**永久不可用**（`new_group` 本身是它上面的集合操作），所以整体销毁、在幸存者上重建。控制类集合（停止协商、checkpoint gather、recovery gather）跑在它上面 |
| 训练组 | GPU=NCCL / CPU=Gloo | 每次故障全体销毁重建：每 active stage 一个 TP 组、每个 pipeline hop 一个**并集组**（`plan.boundary_hops`）、一个覆盖计划放置的全部 rank 的 DP 组 |
| store | `TCPStore` | 托管在 **launcher / Supervisor**（不是 rank 0），全程存活——唯一一条死进程堵不住的通道 |

`GROUP_TIMEOUT = 300s`：故障模型里没有任何东西会碰到它（死 rank 是带外检测、在下次集合之前就知道），
碰到它就说明某个 rank 卡住了，把「永远 hang」变成「抛错」。

### 故障是谁发现的

**launcher**。它是起进程的父进程，唯一依据是操作系统报告的**子进程退出状态**——没有心跳、没有超时猜测、
不可能误判活着的 rank。每轮迭代边界上所有存活 rank 在 store 上会合（`membership.boundary` /
`ControlPlane.observe`），Supervisor 收完尸再公布本轮成员表；被杀的 rank 从来没到过会合点，
所以幸存者是在**发起下一次训练集合通信之前**就知道它没了——**NCCL 永远不会拿到死对端**。

### 一致停止

六条停止码；任何一条被任何幸存 rank 观察到，**所有幸存 rank 抛同一个 `ConsistentStop`**。
协商（`ControlPlane.agree`）跑在**刚重建好的 world 组**上（旧的含死进程，一次集合都跑不了），
原因取自 gather 到的列表**而不是本地视角**，且发生在**建任何训练组之前**。
所以没有人半路继续、不留半完成的计划或组、故障前 checkpoint 原封不动。

| 停止码 | 含义 | 来源 |
|---|---|---|
| `no_feasible_tp` | 没有可行 TP degree | planner 抛出，原样透传 |
| `no_executable_pp` | 没有可执行 PP 分区 | planner |
| `no_feasible_dp_target` | DP 找不到满足显存的目标 | planner |
| `checkpoint_unusable` | 缺失/损坏/摘要不匹配 | 控制面观察 |
| `plan_disagreement` | 各 rank plan digest 不一致 | 控制面观察 |
| `state_mismatch` | 状态重切后完整逻辑张量对不上 | 控制面观察 |

**特例**：`no_executable_pp`（所有 replica 全灭）在真 kill 下等于一个进程都不剩——没有 rank 能观察它、
协商它、正常退出。它因此**不是 rank 可观察的停止，而是作业结束本身**：launcher 汇总退出码，
最后一次完成迭代的 checkpoint 留在盘上可重载。planner 侧这条不可行原因仍由纯函数测试锁定。

---

## 4. 规划层（`planner/` + `plan.py`）

### TP（`planner/tp.choose_tp`）

候选集 `K = { k | k_min ≤ k ≤ |G'|, k = 2^q }`，且必须同时满足：

- attention heads 与 MLP 分片维度能被 k 整除；
- **vocab size 能被 k 整除**——embedding 与 LM head 是 vocab-parallel，漏了这条 planner 会发布一个
  runtime **建不出来**的布局（纯规划测试可以不传 vocab，只校验维度约束；给了 memory_budget 则必须传）；
- 解析显存模型容纳得下（给了 budget 时在 TP 阶段就拒掉过大的 degree，而不是伪装成 DP 失败）。

因 pᵢ≡1，论文的 `argmax k·min(pᵢ)` **退化为：取最大可行 degree，成员按 rank 升序**。
没有 pᵢ 评分分支，没有多套 fallback——不要「补回」这些分支。

### PP（`planner/pp.repartition_pp`）

- 入参 `new_tp_degrees` 是每个既有 stage 的**新 TP degree**，**不是**目标层数（最常见误读）。
- 目标层数 `L_target = floor(L_old × TP_new / TP_old)`；有 TP 组的 stage 至少 1 层；
  TP 组全灭的 stage 为 0 层（stage 被清空，其层数移交同 replica 的幸存 stage——单个死 stage 不会废掉整个 replica）。
- 调整到守恒原模型总层数：正差值逐层给「层数 / TP degree」比值**最小**者，平局按 stage ID **升序**；
  负差值从比值**最大**者收回，平局按 stage ID **降序**。比例比较用**整数交叉乘法**，不引入浮点。
- 返回前强制验证：每 stage 是连续区间、所有全局 layer 恰出现一次、总层数守恒。

### DP（`planner/dp.assign`）

- 先生成新拓扑，再按新拓扑的**可达容量做容量比例静态划分**；映射是纯函数，对同输入幂等。
- 迁移前用**同一个** `memory.memory_feasible` 判 `MemoryFeasible`，不可行就按确定顺序试下一候选，
  全不可行则判计划不可行（`no_feasible_dp_target`）。
- 必须保证：每个 micro-batch 每个 stage **恰好执行一次**；前向 activation 发给**实际的**下游 executor、
  反向 gradient 回到**实际的**上游 executor（邻居可能在另一个 replica，这正是重路由跨 replica 的方式）；
  executor 换人**不影响** global batch 的梯度归一化；不同 replica 的 PP 分层与 TP degree 可以不一样。

### ExecutionPlan（`plan.py`）

`build_plan` 把 TP → PP → DP 串成一个不可变、带版本号、带规范化 digest 的计划，内容包括：
活跃/失效 ranks、每个 `(replica, stage)` 的 `StagePlan`（TP degree / 成员 / 连续 layer 区间）、
每个 `(micro_batch, stage)` 的 `DPPlacement`（executor_ranks）、以及 `state_routes`。

`StateRoute` 是**计划发出的可执行恢复指令**，不是描述：一个 route 覆盖一个 **state group**——
要么一个全局 layer（`layer` 有值、`boundary` 为空），要么 `BOUNDARY_GROUPS = ("embedding","head")`
之一（`layer=None`，跟随 replica 的首/末可执行 stage）。字段含义：

- `donor_kind ∈ {prev_owner, peer_replica, checkpoint}`，`reshard ∈ {peer_copy, gather_reshard, checkpoint_restore}`；
- `donor_degree` / `target_degree` 是读出与写入所用的 TP 布局；`donor_degree == 0` **当且仅当** checkpoint 恢复。

`assert_invariants(plan, previous)` 在发布前锁死结构不变量。`boundary_hops(plan)` 从 assignment 里
读出真实存在的每一跳，给出**两侧 stage 全部 executor_ranks 的排序并集**作为该跳的进程组。

---

## 5. 执行层（`parallel/`）

### TensorParallelStage（`parallel/tp.py`）

一个 stage 拥有**连续的全局 layer id 子集**；replica 的首个可执行 stage 兼管 token/position embedding，
末个兼管 final LayerNorm 与 LM head。分片布局（也是 `memory.py` 预算的、`reshard.shard_dims` 命名的那套）：
Q/K/V 与 `fc1` 按输出（head）维切，`out_proj` 与 `fc2` 按输入维切，token embedding 与 LM head 按 vocab 切，
两个 LayerNorm 与 position embedding 复制。每个分片是**真实叶子参数**，有自己的 grad 与 AdamW 矩状态。

两个 Megatron 集合：`f` = 前向恒等 / **反向 all-reduce**（包住列并行区域 Q/K/V、`fc1` 的输入）；
`g` = **前向 all-reduce** / 反向恒等（归约行并行区域 `out_proj`、`fc2` 与 vocab-parallel embedding 的部分和）。
LM head 的 shard logits 做 all-gather 成完整 logits（反向 split），loss 就是普通交叉熵。
因为 hidden state 在各 rank 上逐字节相同，复制型参数的梯度天然一致，无需额外通信保持同步。

stage 只能从 `local_state`（**本 rank 的分片**，键为稳定逻辑名，正是 `reshard_tp_state` 的返回）构造——
永远从当前计划给它的布局建，绝不从旧布局建。**resharding 不在这里**。

### PipelineRuntime（`parallel/pp.py`，唯一运行时）

执行 warmup 前向 → 1F1B → cooldown 反向，micro-batch 梯度累积成**一次** AdamW 更新。
所有拓扑信息（本 rank 跑哪些 micro-batch、是哪个 stage、pipeline 有几 stage、邻居是谁）
都从当前 plan 的 `DPAssignment` 读。

**边界 scatter/gather**（最近改动最大的地方；旧文档里的 "leader→leader + TP broadcast" 已作废）：
stage 的激活（及其输入梯度）在自己 TP 组内是**复制的**，所以边界只需搬**一份权威副本**——
既不是 per-rank 求和（会把梯度翻倍），也不是只给一个 rank（其他 rank 会饿死）。
按论文 Fig 7 的 P2P 优化，这份副本被切成 `N = max(TP_send, TP_recv)` 个等长连续 chunk，
chunk `k` 由 `up_members[k*U//N] → down_members[k*D//N]` 单独 P2P 传输（`scatter_routing`），
接收方在自己 TP 组内做 intra-node **all-gather** 重建整张量。每对 rank 各承担 `1/N`，
慢速跨 stage 链路总量仍是一份副本，但摊到 N 条并行链路上。异构 TP degree 的边界也是靠它工作的；
两个 TP1 stage 得到两 rank 并集，与旧的 leader pair 完全相同，所以 TP1 跳没有行为变化。

**两处必须融合的收发**：稳态里的 `send_forward + recv_backward` 与 `send_backward + recv_forward`
必须作为一次 `torch.distributed.batch_isend_irecv` 发出。这不是优化：每个 stage 的传输在自己流上有序，
分开发会死锁（一个 stage 阻塞在对端无法匹配的 send 上，而对端也正阻塞在自己的 send 上）。
又因为 NCCL 的 batched P2P 跑在**组自己的 collective communicator** 上（组内每个 rank 都得按同样顺序发起），
所以**一跳的所有 chunk 都骑同一个「两侧 stage rank 并集」组**（`boundary_hops` 命名、控制面构建）。
未融合的跳也用同一个组，理由相同。

### DP 执行（`parallel/dp.py`）

`executor_route(assignment, rank)` 是纯查表：给出本 rank 每个 `(micro_batch, stage)` 的实际上下游
executor ranks（pipeline 两端为 `None`）。`dp_combine_gradients` 在**完整逻辑张量**上跨 replica 求和
（重建各 replica 的张量 → 求和 → 按各自布局重 chunk），复用 T11 的那一条 reshard 路径，
**不是第二次 all-reduce**——这样 PP 分层与 TP degree 不同的 replica 也能统一合并。
`ActivationLog` 记录激活从前向产生到匹配的反向退休期间一直计入显存。

### 重切（`parallel/reshard.py`）

计划 3.3 固定的唯一恢复路径，五步：

```
1 从健康 rank 仍持有的分片收集完整逻辑张量（同一逻辑层的分片也存在于对等 DP replica 上）
2 仅当某 shard 索引在所有健康 rank 上都缺失 → 回落故障前 checkpoint（唯一兜底）
3 按新 degree 重 chunk param / exp_avg / exp_avg_sq
4 分发给新组各 rank
5 恢复出的完整逻辑状态与 checkpoint 逐张量校验，通过后才允许有人继续
```

`reconstruct_full` 是纯函数（donor 与 checkpoint 两条分支都能无进程组单测），
`reshard_tp_state` 只是它上面一层薄的 `all_gather_object`。全程没有 grad。

---

## 6. 恢复与验收（`recovery.py` / `verify.py`）

`recovery.recover` 是**纯编排**，而且编排的是计划已经决定好的事：planner 发 `StateRoute`
（哪个 state group、从哪个 donor、源布局、目标布局），`recover` 只负责执行。
运行时**不再有第二套路由策略**，只保留计划做不了的安全检查，三者都抛错走一致停止：

- donor 集合**真的**不完整才落 checkpoint；
- route 声明的名字必须真的到货；
- 该 stage 应有的名字必须被全部覆盖。

`verify.py` 是原则 A 两个契约的唯一定义：`checkpoint_mismatches / assert_equals_checkpoint`
（`torch.equal`，逐张量）与 `reference_steps / steps_from_anchor / compare_shards /
assert_matches_reference`（重结合带比较）。

**数值口径**：checkpoint 比对用 `torch.equal`；训练数值走 `rtol=1e-4 / atol=1e-5`——TP all-reduce 与
微批切分改变了 FP32 累加顺序。梯度量级低于 `atol` 的分量**不构成**逐分量相对验收。
TP degree 1 时所有集合退化为 no-op、匹配精确；micro-batch 数为 1 时与单进程参考也精确相等。

---

## 7. 入口与配置

```bash
python3 -m resihp.launch --config configs/train.json --failures configs/failures.json
```

入口**不是 `torchrun`**：elastic agent 见到一个 worker 被信号杀死就会连带杀掉/重启其余 worker，
而这里要的是幸存者原地重配。`resihp.launch` 托管 store、按 world size 起 worker、
**子进程死掉时不动其余进程**、汇总每个进程退出码（被杀的应为 -9）。`resihp.train` 是它起的 **worker**；
不带 `RANK` 环境变量直接跑 `resihp.train` 会退化成不依赖 torch 的配置回显，便于单独检查 CLI 与配置文件。

`configs/train.json` 字段是**封闭集合**（少一个报缺失，多一个报未知字段，不做任何默认值兜底）：

```json
{"model_dim":128,"num_layers":6,"num_heads":8,
 "batch_size":8,"micro_batch_size":2,"seed":1234,
 "tp":2,"pp":2,"dp":2,"iterations":8,"memory_budget_bytes":null}
```

约束：`model_dim % num_heads == 0`、`batch_size % micro_batch_size == 0`、world size = `tp × pp × dp`。
`memory_budget_bytes` 是唯一可选字段（`null` = 不设上限），也是在真实运行里打开显存门禁的唯一开关。

`configs/failures.json` 只允许 `after_iteration`（严格递增）与 `failed_rank`（world size 内、不重复）。
**出现速度/降级/检测类字段直接拒绝并报出字段名**——这是守卫，不是待实现功能。
每个 event 恰好使一个新 rank 永久失效，独立触发一次完整重规划与恢复。

输出：每个存活 rank 打印一行

```json
{"acceptance": {"rank": 0, "device": "cuda", "training_backend": "nccl",
  "trained_iterations": [1,2,3,4,5,6,7,8], "plan_versions": [0,1,2,3,4,5],
  "plan_digests": ["..."], "failed_ranks": [1,4,5,6,7]}}
```

一致停止的 run 打印 `{"stopped": "<code>", "reason": "..."}`。
退出码本身证明不了「故障真的发生过而且训练继续了」，**这行 JSON 才是可从外部验证的证据**。

---

## 8. 默认配置的计划演化（回归时的期望值）

默认配置排了**五次连续 fail-stop**，故意让三个维度都真的动起来。
（层数取 6 而不是 4 是有原因的：4 层时重分层恰好把原层数还给每个 stage，就只看得到 TP 变化。）

| 版本 | 生效轮次 | 本次失效 | replica 0 | replica 1 | micro-batch 归属 |
|---|---|---|---|---|---|
| v0 | 1–2 | — | s0 TP2(0,1) L[0-2] / s1 TP2(2,3) L[3-5] | s0 TP2(4,5) L[0-2] / s1 TP2(6,7) L[3-5] | r0=[0,1] r1=[2,3] |
| v1 | 3 | 1 | s0 **TP1**(0) **L[0-1]** / s1 TP2(2,3) **L[2-5]** | 不动 | 不变 |
| v2 | 4 | 4 | 不动 | s0 **TP1**(5) **L[0-1]** / s1 TP2(6,7) **L[2-5]** | 不变 |
| v3 | 5 | 5 | 不动 | **s0 清空**，s1 TP2(6,7) **L[0-5]** | 不变 |
| v4 | 6 | 6 | 不动 | s1 **TP1**(7) L[0-5] | 不变 |
| v5 | 7–8 | 7 | 不动 | **整个 replica 消失** | **r0=[0,1,2,3]** |

TP 在 v1/v2/v4 降 degree，PP 在 v1/v2 搬层、v3 清空 stage，DP 在 v5 真正重路由 micro-batch。
各 rank 实际执行轮次：rank 1 = `[1,2]`、rank 4 = `[1,2,3]`、rank 5 = `[1..4]`、rank 6 = `[1..5]`、
rank 7 = `[1..6]`、rank 0/2/3 = `[1..8]`。

---

## 9. 明确不做（提改动前先看这张表）

| 项 | 说明 |
|---|---|
| 自动故障检测 / 心跳 | 故障来自确定性故障表，不是运行时检测 |
| 多卡同轮失效 | 每个 event 恰好一个 rank（计划锁定） |
| fail-slow / 速度启发式 / standby | pᵢ≡1，相关构造有全仓扫描门禁要求命中为 0 |
| 多节点跨机 | 单机 8 卡 |
| 集合通信 mid-flight 恢复 | 只在安全点（迭代 + AdamW step 完成后）重配 |
| 跨安全点梯度累积 | 一旦引入，`grad` 的恢复契约必须先改（见不变量 2） |
| 性能优化 | 边界只做功能正确；scatter/gather 是论文结构，不是为了跑分 |
| 混合精度 / 多优化器 | FP32 + AdamW 固定 |
| 安装 / 升级 / 修改 Torch、CUDA、NCCL | 镜像内已就绪，不动 |

---

## 10. 当前状态与交接注意

- 开发机（Windows，无 torch）：`python -m pytest -q` → **125 passed, 12 skipped**；12 项 skip 全是
  分布式/GPU 模块在收集阶段整模块跳过（`torch` 不可导入），不是被跳过的断言。
- 8 卡目标机：最近一轮全量审阅改动了真实通信结构（**每跳并集组、边界 scatter/gather、1F1B 接入生产、
  recovery 改为执行 `state_routes`**），**分布式与数值门禁需要在目标机复跑一遍**。
  旧的目标机确认是针对已经不存在的代码取得的，**不能沿用**。
- 分支 `fix/dp-replica-level-rerouting`。逐条自检见 [ACCEPTANCE.md](ACCEPTANCE.md)，
  每轮修改的根因见 [PROGRESS.md](PROGRESS.md)（进度只记在这一个文件，不在根目录另建进度文档）。
- 本项目不宣称证明所有硬件绝对无误；结论建立在确定性数值对照、状态不变量、故障原子性与多进程端到端之上。
