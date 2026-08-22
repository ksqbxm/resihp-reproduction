# ResiHP 结项自检报告（T18）

对照计划文档「五、交付顺序」与「六、假设与完成标准」逐条自检。

## 0. 报告口径

- 每条结论都标出**证据在哪、由谁产生**。凡本机（Windows，**无 torch**，计划硬性禁止安装/升级 torch）
  跑不了的，一律写「待 8 卡目标机执行」，不以「实现看起来对」代替执行结果。
- 本机可执行部分：`python -m pytest -q` → **117 passed, 12 skipped**。12 项 skip 全部是分布式 / GPU
  模块在收集阶段整模块 skip（`torch` 不可导入），不是被跳过的断言。
- 本报告不复述各任务的实现细节，那些在 `docs/PROGRESS.md`；这里只回答「完成标准成立没有、凭什么」。

## 1. GPU/NCCL 验收（完成标准：≥2 次连续 fail-stop 并继续）

验收命令即计划「一」规定的唯一入口，从仓库根目录执行：

```bash
torchrun --standalone --nproc_per_node=8 -m resihp.train --config configs/train.json --failures configs/failures.json
```

默认配置是 `TP2 × PP2 × DP2 / 6 层 / 8 轮`，故障表排了**五次连续 fail-stop**（每次单 rank，
`after_iteration` 严格递增、rank 不重复），最后一次之后仍有 2 轮训练。层数取 6 而不是 4 是有原因的：
4 层时 `repartition_pp` 恰好把原来的层数还给每个 stage，验收就只能看到 TP degree 变化；6 层才让
**同一个事件既降 TP degree 又把一层搬过 stage 边界**。故障序列也不再是两次就收手——只有把整个
replica 1 打空，micro-batch 归属才会真的换 replica，这才是**真实 DP reroute**。

纯 planner 对这套配置给出的计划序列（本机实跑 `build_plan` 核对过；`resihp/train.py` 里
`build_initial_plan` 用同一套入参，所以 digest 逐版相同）：

| 版本 | 生效轮次 | 失效 rank | active ranks | replica 0 stages | replica 1 stages | micro-batch 归属 |
|---|---|---|---|---|---|---|
| v0 | 1–2 | — | 0–7 | s0 TP2 (0,1) L[0,1,2] / s1 TP2 (2,3) L[3,4,5] | s0 TP2 (4,5) L[0,1,2] / s1 TP2 (6,7) L[3,4,5] | r0=[0,1] r1=[2,3] |
| v1 | 3 | 1 | 0,2,3,4,5,6,7 | s0 **TP1** (0,) **L[0,1]** / s1 TP2 (2,3) **L[2,3,4,5]** | 不动 | r0=[0,1] r1=[2,3] |
| v2 | 4 | 1,4 | 0,2,3,5,6,7 | 不动 | s0 **TP1** (5,) **L[0,1]** / s1 TP2 (6,7) **L[2,3,4,5]** | r0=[0,1] r1=[2,3] |
| v3 | 5 | 1,4,5 | 0,2,3,6,7 | 不动 | **s0 清空**；s1 TP2 (6,7) **L[0..5]** | r0=[0,1] r1=[2,3] |
| v4 | 6 | 1,4,5,6 | 0,2,3,7 | 不动 | s1 **TP1** (7,) L[0..5] | r0=[0,1] r1=[2,3] |
| v5 | 7–8 | 1,4,5,6,7 | 0,2,3 | 不动 | **整个 replica 消失** | **r0=[0,1,2,3]** |

三个维度都真的动了：**TP** 在 v1/v2/v4 各降一次 degree；**PP** 在 v1/v2 搬层、在 v3 清空一个 stage
并把全部 6 层交给幸存 stage；**DP** 在 v5 把 micro-batch 2、3 重路由到 replica 0。
`tests/test_acceptance.py::test_the_shipped_schedule_exercises_tp_pp_and_dp` 把这三条写成门禁，
且不需要 GPU——计划序列是两个配置文件的纯函数。

每个 rank 实际应执行的轮次：rank 1 = `[1,2]`，rank 4 = `[1,2,3]`，rank 5 = `[1,2,3,4]`，
rank 6 = `[1..5]`，rank 7 = `[1..6]`，rank 0/2/3 = `[1..8]`。

**自动门禁**：`tests/test_acceptance.py::test_torchrun_nccl_acceptance`（需 8 张 GPU，不足即 skip）。
它用 `sys.executable -m torch.distributed.run`（`torchrun` 控制台脚本就是它的薄包装）在仓库根目录发起
上面那条 argv，然后只看 job 自己吐出的证据：

| 验收要求 | 门禁检查的事实 |
|---|---|
| 跑完而不是中途一致停止 | 启动器退出码 0；stdout 里没有 `{"stopped": ...}` 行 |
| 真的在 GPU / NCCL 上 | 每个 rank 报告 `device == "cuda"` 且 `training_backend == "nccl"` |
| 每次故障恰好一个新计划、各 rank 一致 | 各 rank 的 `plan_versions == [0..5]`，且 `plan_digests` 八个 rank 逐版相等，**并等于纯 planner 对同一配置算出的 digest** |
| 五次 fail-stop 都真的生效 | 各 rank 报告的 `failed_ranks == [1,4,5,6,7]` |
| 失效 rank 永久退出、幸存者继续训练 | 各 rank **实际执行**的轮次列表逐 rank 等于上表的 active 集合——每个失效 rank 停在自己那次事件，rank 0/2/3 跑满 8 轮 |
| 安全点原子提交了 checkpoint | 运行前先清掉 `checkpoint.pt`，跑完后它存在、且没有 `checkpoint.pt.tmp` 残留 |

为让这条命令**从外部可验证**，`resihp/train.py` 在跑完时每个 rank 打印一行
`{"acceptance": {...}}`（rank / device / backend / 实际执行的轮次 / 计划版本与 digest / 失效 rank）。
这是本任务对 `resihp/` 的**唯一**改动：退出码本身证明不了「两次故障发生过且训练继续了」。

**离线验证（本机已执行）**：用真实 planner 对真实配置合成一份健康 run 的 stdout 喂给门禁自身的断言，
健康流水通过；再逐一注入 18 种缺陷——少报一个 rank、重复上报、落在 CPU、训练组是 Gloo、两次故障只出一个
计划、版本重复、某个 rank digest 不同、八个 rank 一致但不等于 planner、第二次故障没生效、失效 rank 继续
训练、失效 rank 多跑一轮、幸存者最后一次故障后停训、幸存者漏一轮、出现一致停止行、退出码非零、没有
checkpoint、残留 `.tmp`、以及配置层面的「最后一次故障落在最后一轮」——**18/18 全部被捕获**，不存在写了
但永不触发的检查。

**状态：待 8 卡目标机执行。** 本机无 torch、无 GPU，这一条只能在目标机上真跑一次才算数。

## 2. 全仓扫描（完成标准：禁用构造为零）

门禁 `tests/test_acceptance.py::test_no_banned_constructs`，**不依赖 torch，本机每次都跑**。
扫描范围是「代码」：`resihp/**/*.py` + `configs/*.json` + 仓库根目录的 `*.py`。

| 禁用项 | 匹配式（忽略大小写） | 命中 |
|---|---|---|
| Detector | `detector` | 0 |
| pᵢ | `\bp_i(?![a-z])`、`p_{i}`、`\bpi\b`、`pᵢ` | 0 |
| 速度 / 降速分支 | `\bspeed`、`slowdown`、`slower`、`fail.?slow`、`nvidia.?smi`、`heartbeat` | 0 |
| standby | `standby` | 0 |
| Algorithm 1 | `algorithm\s*1(?!\d)` | 0 |
| 旧入口 | `hello_dist`、`nccl_test` | 0 |
| 前向兼容层 | `\blegacy`、`\bdeprecated`、`backward.?compat`、`forward.?compat` | 0 |

外加：`hello_dist.py` / `nccl_test.py` 这两个文件名在**整个仓库**任何位置都不存在（`rglob` 检查）。
`git ls-files` 也确认根目录只有 `CLAUDE.md`、`RESIHP_TASKS.md`、`.gitignore` 与四个目录，没有任何旧脚本。

匹配式的**检出力已验证**（本机跑过变异核对）：18 条典型违规写法（含 `legacy_layout`、`p_i_score`、
`speed_ratio`、`Algorithm1` 这类标识符形态）全部命中；11 条容易被粗糙匹配式误伤的真实代码行
（`_step_int`、`tp_index`、`p_int`、`pipeline`、`algorithm 10` 等）全部不命中。

**人工复核了扫描范围之外的三处近似命中，均非违规**：

- `tests/test_config.py` 把 `speed` / `p_i` / `detector` 作为参数化用例——它们出现在那里正是因为加载器**必须
  拒绝**这些字段，是守卫而不是实现；
- `tests/test_parallel_reshard.py` 的 `"degrade"` 指 **TP degree 减半**，与速度无关；
- `resihp/parallel/reshard.py` 的 "checkpoint-fallback" 是计划 3.3 明文规定的恢复分支（健康 replica 全缺失
  才读 checkpoint），`resihp/parallel/pp.py` 的 "no old-layout compatibility is kept" 是一句**否定**陈述。

`tests/` 之所以不在自动扫描范围内：那里唯一的命中就是上面第一、二条，把它们纳入就必须按文件名开白名单，
而白名单会在测试改名的那一刻失效。范围写在门禁的 docstring 里，不靠读者猜。

## 3. A–F 测试对照

| 组 | 覆盖 | 文件 | 状态 |
|---|---|---|---|
| **A 模块级** | 配置/故障 JSON 校验、显存模型、TP 候选与成员、PP 分层、DP 容量与重路由、ExecutionPlan 与不变量、checkpoint 读写 | `test_config.py`、`test_memory.py`、`test_planner_{tp,pp,dp}.py`、`test_plan.py`、`test_checkpoint.py`、`test_entrypoint.py` | 纯函数部分**本机全绿**；`test_checkpoint.py` 需 torch，待目标机 |
| **B 不变量** | active/assigned 一一对应、层归属唯一连续、micro-batch·stage 恰一次、版本递增与 digest 一致、重路由幂等 | `test_plan.py::assert_invariants` 系列 + `test_end_to_end.py` + `test_fault_sequences.py` 每次重配后强制校验 | 纯函数部分本机全绿；分布式部分待目标机 |
| **C 对比验收** | 恢复前逐张量等于 checkpoint；恢复后与「同 checkpoint 起点 + 新拓扑 + 新配置实际 batch + 同种子」参考一致；多故障点位（每 N / 每 2N / 随机）与多次序列回归 | `test_recovery.py`、`test_end_to_end.py`、`test_fault_sequences.py`（5 条序列 × Gloo/NCCL） | 目标机：T16 两项已确认全过；T17 五条 Gloo 已确认全过，NCCL 待复跑 |
| **D 组合与端到端** | 七项两两组合（均执行真实前反向）× Gloo/NCCL = 14 项；完整 3D `TP2×PP2×DP2` 8 进程 2 项 | `test_combinations.py`、`test_end_to_end.py` | T16 两项目标机已确认；14 项组合待目标机执行 |
| **E 资源耗尽与错误注入** | 六条一致停止条件各一项 + 「健康 donor 全失但 checkpoint 可用」 | `test_recovery.py`（六条）、`test_fault_sequences.py::donor_exhaustion` | 目标机：Gloo 已确认，NCCL 待复跑 |
| **F 显存公式专项** | 两组已知配置逐项手算核对字节数 + 「刚好满足 / 超一字节」边界 | `test_memory.py` | **本机全绿** |

## 4. 完成标准逐条

| 完成标准（计划六） | 结论 | 依据 |
|---|---|---|
| A–F 全部测试通过 | **部分待执行**：无 torch 的 A/F 与 B 的纯函数部分本机全绿；分布式与 GPU 门禁见第 3 节状态列 | `python -m pytest -q` = 117 passed / 12 skipped |
| GPU/NCCL 下 ≥2 次连续 fail-stop 并继续 | **待目标机执行** | 第 1 节：命令、期望计划序列、门禁与 18 项变异核对均已就位 |
| TP/PP/DP 均真实执行（非仅元数据） | 成立（凭已确认的目标机结果） | T16 两项在目标机全过：真实 TP all-reduce 分片前反向、1F1B 层迁移、跨 replica activation/gradient；T15 的七项组合明确要求「执行真实前反向」 |
| 参数 / AdamW / 迭代号 / 数据游标无丢失 | 成立（同上） | 恢复后各分片与 anchor 逐张量 `torch.equal`（含 `exp_avg`/`exp_avg_sq`/`step`）；cursor 逐轮等于 `iteration-1`，恢复后等于 checkpoint 的 `completed_steps`。**`grad` 不在其中**：安全点只出现在 AdamW step 之后、无跨安全点梯度累积，下一轮 `zero_grad` 后重算（计划文档「二·补」） |
| 恢复前精确等于 checkpoint、恢复后与新配置参考一致 | 成立（同上，NCCL 侧 T17 待复跑） | 原则 A 双口径：前半段 `torch.equal`，后半段对「同起点 + 新拓扑 + 新 batch + 同种子」参考 `allclose`（NCCL 上梯度差已降到 ~1e-08–3e-08） |
| 资源不足时全体一致退出、最后 checkpoint 完整 | 成立（Gloo 已确认，NCCL 待复跑） | 六条停止条件各一项门禁；随机序列一路注入到资源耗尽后以 `no_executable_pp` 全体一致退出，故障前 checkpoint 仍可重载 |
| 代码中不存在 Detector / pᵢ / 速度分支 / standby / Algorithm 1 / 旧入口 / 前向兼容 | **成立，本机已执行** | 第 2 节，7 类禁用项命中数全为 0，且匹配式的检出力经变异核对 |

## 5. 待执行清单（8 卡目标机）

按顺序执行，前一条失败就不必往下走：

```bash
python -m pytest -q tests/test_acceptance.py -v
```

```bash
torchrun --standalone --nproc_per_node=8 -m resihp.train --config configs/train.json --failures configs/failures.json
```

```bash
python -m pytest -q
```

第二条是人工验收：屏幕上应出现 8 行 `{"acceptance": ...}`，其中 `trained_iterations` 为
rank 1 `[1,2]`、rank 4 `[1,2,3]`、rank 5 `[1,2,3,4]`、rank 6 `[1..5]`、rank 7 `[1..6]`、
rank 0/2/3 `[1..8]`，八行的 `plan_digests` 完全相同（各 6 项），且不出现 `{"stopped": ...}`。
第一条门禁检查的就是这些，人工跑一遍是为了留下可读的验收记录。

全部通过后，把本报告第 1 节与第 4 节的「待目标机执行」改为「已验证」，并在 `docs/PROGRESS.md` 记录结果。

## 6. 已知边界（记录，不掩盖）

- 数值口径：TP all-reduce 与微批切分相对参考的整批一次前反向改变了 FP32 累加顺序，因此除
  checkpoint 比对用 `torch.equal` 外，训练数值用的是 `rtol=1e-4 / atol=1e-5` 的重结合带；
  梯度量级低于 `atol` 的分量不构成逐分量相对验收（`docs/PROGRESS.md` 有详述）。
- `resihp/model.py` 模块级钉死 `torch.set_float32_matmul_precision("highest")`：TF32 是进程级默认且各
  torch 版本变过，计划要求全程 FP32，这条数值契约必须显式声明而不是继承默认。
- 故障粒度：每个事件恰好一个 rank（计划锁定），不做多 rank 同轮批量生效。
- 本报告不宣称证明所有硬件绝对无误；结论建立在确定性数值对照、状态不变量、故障原子性与多进程端到端之上。
