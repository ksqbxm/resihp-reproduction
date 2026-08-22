# ResiHP Fail-stop 动态 3D 并行恢复 — 最终实施计划

> 故障粒度：**每次单 rank 逐次处理**（每个 fail-stop 事件独立触发一次完整重规划与恢复；不做多 rank 同轮批量生效）。

---

## 一、目标与约束

在 `/data/ubuntu/resihp-project` 用原生 PyTorch 实现一个可真实训练的微型 decoder-only Transformer，并实现 ResiHP 的**纠错恢复**路径：接收确定性 fail-stop 事件，依次完成 TP→PP→DP 重规划、通信组重建、训练状态迁移，并从原进度继续训练，支持累计、反复注入故障。

**范围约束（硬性）：**

- 只实现纠错恢复，**不实现** Detector、心跳、硬件测速、fail-slow、时间预测、性能优化。
- 只处理 **fail-stop**；失效 rank 永久“逻辑下线”（仍参与控制面同步，不再参加训练通信与计算）。
- 所有存活设备均视为健康，故障速度概念不存在，**pᵢ 恒等于 1，不作为任何输入**。
- 不使用 Megatron；不安装/升级/卸载/修改 Torch、CUDA、NCCL。
- 默认拓扑 `TP=2, PP=2, DP=2`，8 进程；GPU 用 NCCL，CPU 测试用 Gloo，**共用同一 Scheduler、状态格式与执行路径，仅切后端**。
- FP32、固定随机 token、next-token cross-entropy、AdamW 为唯一优化器。
- 写入仅限项目目录；不执行 Docker、系统清理、`rm -rf`。

**唯一入口（删除 `hello_dist.py`，不保留旧入口/兼容层/双路径）：**

```bash
torchrun --standalone --nproc_per_node=8 -m resihp.train \
  --config configs/train.json --failures configs/failures.json
```

**故障文件（只含 fail-stop，按迭代严格递增，rank 合法不重复；出现速度/降级/检测字段直接拒绝）：**

```json
{"events": [
  {"after_iteration": 2, "failed_rank": 1},
  {"after_iteration": 4, "failed_rank": 5}
]}
```

> 每个 event 恰好使**一个**新 rank 永久失效；多个事件可分布在多个安全点，直到资源不可行。**每个事件独立触发一次完整重规划与恢复。**

**每次写代码前必须执行 `CLAUDE.md` 的约束检查**（最小必要实现、每步明确验证、只改本任务相关内容、不留兼容层）。

---

## 二、两条锁定的根本原则（不可动摇）

### 原则 A — 数值验收两段式（替代“与无故障参考一致”）

1. **恢复前**：重配完成、继续训练之前，完整逻辑参数与 AdamW 状态（`param`/`exp_avg`/`exp_avg_sq`/`step`）与故障前最近 checkpoint **逐张量精确相等**（`torch.equal`，dtype 一致）。
2. **恢复后**：与“**同一 checkpoint 起点 + 新拓扑 + 新配置实际 batch + 同一随机种子**”重建的参考运行**逐步一致**（确定性设置下全精度一致，不用宽松容差）。

> 明确：参考基准是“新配置从同起点”，**绝不**是“无故障从头”。踢 rank 后 batch 与归约结构改变，与无故障参考必然分岔——那是语义不同，不是精度误差。

### 原则 B — 纯 fail-stop 确定性重路由（替代 Algorithm 1）

- 删除 pᵢ 进度启发式、`P_{d,i,t}`、“pending 最多/可接收最强”、“模拟到达时间”、δ 阈值、standby——这些是 fail-slow 机制，pᵢ≡1 下空转。
- DP/数据重路由是**纯函数**：`(step, failure_signature, active_topology) → assignment`，同输入必同输出；仅依据当前活跃拓扑与故障时刻状态，结果由顺序/ID 决定。
- **故障粒度：每次单 rank 逐次处理。** 每个 fail-stop 事件独立触发一次完整重规划与恢复；`failure_signature` 由“到当前事件为止已累计失效的 rank 集合”唯一确定。**不做多 rank 同轮批量生效、不做跨轮迁移/抢占。**

---

## 二·补 — 恢复状态的边界：`grad` 不是持久状态

> 本节是对 3.3 / 3.4 中「迁移 param/grad/exp_avg/exp_avg_sq」一句的收口修正，以真实执行语义为准。

安全点只出现在「当前迭代完成 → AdamW step 完成 → 原子 checkpoint」之后，且不存在跨安全点的
gradient accumulation：下一迭代必然先 `zero_grad` 再重新 forward/backward。因此安全点时刻驻留的
`param.grad` 是**已经被消费掉的**旧值，搬运它没有任何语义价值。

**持久恢复状态（唯一定义）**：

```text
param
exp_avg
exp_avg_sq
step
iteration（= 数据游标）
RNG
```

`grad` 既不进 checkpoint，也不进 `ExecutionPlan.state_routes`，也不进任何 reshard/迁移传输；
下一迭代重新计算。若将来引入跨安全点的梯度累积，本节必须先改。

---

## 三、实现方案（附实现细节）

### 3.1 确定性训练基线

- 自建：token/position embedding、LayerNorm、多头因果自注意力、MLP、残差、LM head。
- **每个 Transformer layer 用稳定全局 ID**，使 TP 分片、PP owner 变化后仍能唯一定位参数（重规划前后全局 layer ID 与逻辑参数名保持稳定）。
- 固定模型初始化、输入 token、数据顺序、RNG 状态。
- 先实现**单进程参考训练**，记录每轮输入、loss、参数摘要、优化器摘要。
- checkpoint 保存：完整逻辑模型、AdamW（`exp_avg`/`exp_avg_sq`/`step`）、迭代号、数据游标、CPU/CUDA RNG 状态、执行计划版本。
- **测试**：固定种子下输入/loss/一步更新可重复；save/load 后下一步与不中断训练一致；参数缺失/形状错/游标不一致能明确失败。

### 3.2 执行计划与控制面

- 唯一不可变、带版本号 `ExecutionPlan`（**先记录最小必要集，跑通再加**），含：活跃/失效 ranks；每个逻辑 DP replica、PP stage 的 TP 成员；每 stage 连续 layer 范围；每个 micro-batch·stage 的执行 rank；参数/优化器状态的 donor、目标与重分片方式；计划版本与规范化摘要；不可行时的结构化原因。
- **通信 rank / TP shard / PP owner 只由当前 ExecutionPlan 决定，禁止从旧布局隐式推导。**
- 所有进程持有一个**始终存活的 Gloo 控制组**；训练组按当前计划建（GPU=NCCL / CPU=Gloo）。
- **每次故障安全点严格顺序执行**（单 rank）：
  1. 完成并提交当前迭代。
  2. 原子保存最近完成迭代 checkpoint。
  3. 广播 fail-stop 事件。
  4. 标记该 rank 永久失效。
  5. TP→PP→DP 重规划。
  6. 所有进程按统一顺序释放旧训练组、建新组。
  7. 恢复/迁移/重切参数与 AdamW。
  8. 校验计划摘要与完整逻辑状态摘要一致。
  9. 从下一迭代继续。
- **测试**：每事件只生成一个新计划版本；失效 rank 永久退出训练路径；各 rank 计划摘要一致。

### 3.3 TP 重规划与真实张量重切

- 排除失效 rank：`G' = G \ F_stop`。
- 候选 degree：`K = { k | k_min ≤ k ≤ |G'|, k = 2^q }`，并满足 attention heads / MLP 分片维度可被 k 整除、解析显存预算容纳（分片参数 + 复制参数 + 梯度 + AdamW 两份矩状态 + 峰值 activation）。
- **因 pᵢ≡1，子组选择退化为：在满足上述约束的候选中取最大可行 degree，成员按 rank 升序确定。**（注：此为论文 `argmax k·min(pᵢ)` 在全健康下的退化形式；**不保留 pᵢ 评分分支、不保留多套 fallback**。）
- 真实 TP：attention-head 与 Q/K/V 分片、MLP column-parallel、attention/MLP output row-parallel、必要 TP all-reduce、embedding 与 LM head 唯一固定分片。
- degree/成员变化时重切：
  1. 从其他健康 DP replica 收集完整逻辑张量；
  2. 某 shard 在所有健康 replica 均缺失时从故障前 checkpoint 恢复；
  3. 按新 degree 重切 `param/exp_avg/exp_avg_sq`（**不含 `grad`**，见下）；
  4. 分发新 TP 组；
  5. gather 后与 checkpoint 完整逻辑状态逐张量校验。
- **异构 TP 边界（功能正确即可，不做 P2P 性能优化）**：前向用 leader gather → 计算 → scatter/broadcast；反向对应 scatter-reduce，确保梯度不重复累加/不丢失。此路径单列测试。
- **测试**：2 的幂候选生成；整除约束；k_min 显存约束；最大可行 degree 与确定性成员；单次/连续掉卡；无可行 TP；TP1/TP2 前反向与一步 AdamW **与单进程参考数值一致**；`TP2→TP1`、TP 成员替换的 gather/reshard 无丢失；**异构 TP 边界反向梯度与参考逐元素一致**；donor 恢复与 checkpoint fallback 分别单测。

### 3.4 PP 重分层与状态迁移

- `new_tp_degrees` 是每个既有 PP stage 的新 TP degree，不是目标层数；目标层数：`L_target = floor(L_old × TP_new / TP_old)`；有 TP 组的 stage 至少 1 层，TP 组完全失效的 stage 为 0 层。旧布局可包含已清空 stage，以支持连续重分层。
- 调整初始目标层数以守恒原模型总层数：正差值逐层分配给当前“层数/TP degree”最小者，平局按 stage ID 升序；负差值从该比值最大者逐层收回，平局按 stage ID 降序，且 active stage 至少保留 1 层。比例比较使用整数交叉乘法，避免浮点溢出。
- 每 stage 连续区间；所有全局 layer 恰好出现一次，连续、不重叠、不遗漏，且返回前必须验证总层数守恒。
- 迁移 layer 一并迁移 `param/exp_avg/exp_avg_sq/step` 与元数据（**不含 `grad`**，见下）；接收 stage TP degree 不同则**直接按目标布局重切，不留旧布局兼容**。embedding/LM head 归属首/尾可执行 stage。
- 运行时由计划生成 Forward/Backward/Send/Recv/WeightUpdate 原语，统一 **1F1B 功能调度**（内部保留 F/B/W 三类）。
- **测试**：论文示例 `(4,4,4)→(5,2,5)`；奇数余层与确定性余数；首/中/末 stage 故障；stage 清空；连续多次重分层；layer 连续唯一完整；embedding/LM head owner；每层所有训练状态迁移前后逐张量一致；PP 两阶段前反向与一步更新**与参考一致**；TP 重切 + PP 移层组合验证 shard/owner/通信边界一致。

### 3.5 DP 确定性重路由（替代 Algorithm 1）

- 保留逻辑 DP replica 的数据归属；每次故障后**先生成新拓扑，再按新拓扑可达容量做确定性 batch 划分（容量比例静态重路由）**。
- 微批映射为纯函数 `(step, failure_signature, active_topology) → assignment`，同输入必同输出，不依赖运行时延迟/进度。
- 失效 stage 的 micro-batch → 按确定规则（replica ID 升序等）重路由到健康 peer stage。
- **故障粒度：每次单 rank 逐次处理。** 每个 fail-stop 事件独立触发一次重规划；**只改本次事件后的新轮任务，不做跨轮迁移/抢占**。
- 迁移前用解析显存模型判 `MemoryFeasible`（与 TP 的 k_min **同一计算器**），不可行则按确定顺序试下一候选，全不可行判计划不可行。
- 必须保证：每个 micro-batch 每个 stage 恰好执行一次；forward activation 发给实际下游 executor、backward gradient 返回实际上游 executor；activation 在对应 backward 完成前持续计入显存；workload 不同时归属源与目标；executor 改变不影响 global batch 梯度归一化；不同 replica 的 PP 分层与 TP degree 可不同。
- **测试**：容量划分与目标排序；失效 stage 全部 micro-batch 被迁出；执行完整性（每 stage 恰一次）；显存允许/拒绝/边界；多目标平局确定性；activation 生命周期；跨 replica 前向 activation 与反向 gradient；不同 PP 分层/不同 TP degree 的 replica 间迁移后**梯度与参考一致**；重路由函数对同输入**幂等**。

### 3.6 状态恢复与失败原子性（单一路径）

- checkpoint **先写临时文件、完整校验后原子替换**；恢复成功前不覆盖最后有效 checkpoint；**只保留当前恢复所需的唯一最新 checkpoint**。
- **唯一恢复链路**：故障 → 读 checkpoint / 从健康 replica 收集 → 状态恢复 → 重排计划 → 建群 → 继续。缺失 shard 优先从健康 replica 收集，全缺失才从 checkpoint 恢复；恢复为完整逻辑张量后**仅按新计划重新分片**。
- **一致停止条件**（任一触发，所有 rank 达成一致后保留故障前 checkpoint、输出结构化根因、正常退出，不发布半完成计划、不允许部分 rank 继续/死锁）：
  - 无可行 TP degree；
  - 无可执行 PP 分区；
  - DP 无满足显存的目标；
  - checkpoint 缺失/损坏/摘要不匹配；
  - 各 rank 计划不一致；
  - 状态重切后完整逻辑张量不一致。
- **测试**：临时文件不完整不替换；各停止条件全部进程一致退出、无半完成组/计划、最后 checkpoint 可重载、报错只指根因不用冗余兜底掩盖。

---

## 四、测试计划（逐模块锁定 + 不变量 + 属性）

**执行纪律**：严格“先写测试 → 最小实现 → 跑通 → 锁定 → 再下一模块”；每模块通过前不进入下一模块；**每次写代码前跑一遍 `CLAUDE.md` 约束**。

### A. 模块级（纯函数优先，穷举/边界）

配置校验、故障 JSON、显存模型、TP 候选与重切、PP 分层与迁移、DP 容量与确定性重路由、checkpoint 读写、通信组重建映射一致与互斥。

### B. 不变量断言（每次重配后强制自动运行）

- `active_ranks` 与 `assigned_ranks` 一一对应、不重叠；`active_ranks` 单调收缩。
- 每个全局 layer 任意时刻恰被一个 stage 拥有（连续、无重叠、无缺口）。
- `Σ microbatches_per_replica == 新配置实际 batch`；每个 micro-batch·stage 恰执行一次。
- 所有张量 shape 与新拓扑一致、可被当前进程组消费；计划版本严格递增、各 rank 摘要一致。
- 重路由函数对同输入幂等。

### C. 对比验收

- 恢复前逐参/逐优化器状态精确等于 checkpoint。
- 恢复后与“新配置参考 run”逐步全精度一致（确定性设置）。
- 覆盖多故障点位（每 N step、每 2N step、随机间隔）与多次序列回归。

### D. 组合与端到端

- 两两组合（须执行真实前反向，不只比计划）：TP+PP、TP+DP、PP+DP、Scheduler+建群、Scheduler+状态迁移、状态迁移+checkpoint、动态通信组+PipelineRuntime。
- 完整 3D：8 进程 Gloo `TP2×PP2×DP2` — 无故障参考 → 第 2 轮后失效 TP rank → TP 重切/PP 移层/DP 重路由 → 恢复 → 第 4 轮后失效另一 DP replica 的 rank → 再次重配恢复 → 训练至结束。校验：无 collective 顺序错/死锁；失效 rank 不再训练；每故障仅一个新计划；**恢复前状态精确等于 checkpoint；恢复后与新配置参考一致**；数据不重不漏；layer 与 micro-batch·stage 不重不漏。
- **随机故障序列（固定 seed）**：随机安全点 + 随机 rank，逐次注入 1~K 个（每次单 rank）直至资源耗尽，全程校验 B 组不变量。
- 反复注入（固定间隔坏一个）验证稳定性与无泄漏。

### E. 资源耗尽与错误注入

TP 候选空；PP 无法覆盖全层；DP 目标显存全不足；健康 donor 全失但 checkpoint 可用；checkpoint 损坏；rank 间计划摘要不一致；状态迁移摘要错误。每场景验证：超时前全体一致退出、无半完成组/计划、最后 checkpoint 可重载、报错只指根因。

### F. 显存公式专项

除边界“刚好满足/超一字节”外，**另用已知配置手算核对**参数/梯度/AdamW/activation 字节数，防止公式细节错。

---

## 五、交付顺序（严格逐步，门禁制）

1. 删 `hello_dist.py` 及一切旧路径；写 Planner 核心纯函数 + 单测（TP 候选、PP 分层、DP 容量、确定性 reroute）。
2. 训练基线（3.1）+ checkpoint + 恢复前/后双口径断言骨架。
3. 接入确定性 reroute 与新 batch 计算 + B 组不变量测试。
4. TP 真实重切（3.3，含异构边界反向）→ PP 迁移（3.4）→ DP 重路由（3.5），逐个锁定。
5. 原子重配与一致停止（3.6）；扩展到随机序列与反复注入。
6. 形成单一运行脚本与测试门禁：`python3 -m pytest -q` 全绿方可进下一阶段；`torchrun --standalone --nproc_per_node=8 -m resihp.train ...` 做 GPU/NCCL 验收（≥连续两次 fail-stop 并继续）。

---

## 六、假设与完成标准

- 容量比例静态重路由 + 按新配置实际 batch（已锁定）。
- **故障粒度：每次单 rank 逐次处理（已锁定）。**
- pᵢ≡1；Scheduler 不读运行时间、不推断故障、不调 nvidia-smi。
- 完成标准：A–F 全部测试通过；GPU/NCCL 下 ≥2 次连续 fail-stop 并继续；TP/PP/DP 均真实执行（非仅元数据）；参数/梯度/AdamW/迭代号/数据游标无丢失；恢复前精确等于 checkpoint、恢复后与新配置参考一致；资源不足时全体一致退出、最后 checkpoint 完整；**代码中不存在 Detector、pᵢ、速度/降速分支、standby、Algorithm 1、旧入口或前向兼容逻辑**。
- 不宣称证明所有硬件绝对无误；以确定性数值对照、状态不变量、故障原子性与多进程端到端共同证明本范围内功能正确。

---

## 附：开工前确认

- **确认 `CLAUDE.md` 内容**：本计划多处要求“每次写代码前执行 CLAUDE.md 约束”，开工前先打开确认其内容，确保与本计划不冲突。
