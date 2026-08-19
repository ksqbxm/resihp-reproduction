# ResiHP — Claude Code 任务拆分与可直接发送的指令

每条 `>` 引用块即为**可整段复制发给 Claude Code 的指令**。一次会话只做一个任务。

## 通用前缀（已写进 CLAUDE.md，下列指令中已省略重复说明）

每条指令都以固定三句开头：读 `CLAUDE.md` → 读计划文档指定章节 → 输出约束检查清单。

---

## 阶段 0：地基

### T0 — 建立约束文件与清理旧路径

> 读 `CLAUDE.md` 全文，再读 `docs/ResiHP_failstop_recovery_plan.md` 的「一、目标与约束」和「五、交付顺序」。输出约束检查清单后开始。
>
> 任务：初始化项目骨架。
> 1. 删除 `hello_dist.py` 及任何旧入口、旧脚本、旧 README 中的旧用法，不保留注释版本。
> 2. 建立包结构：`resihp/`（`__init__.py`、`train.py` 占位）、`configs/train.json`、`configs/failures.json`、`tests/`、`docs/`。把计划文档放到 `docs/ResiHP_failstop_recovery_plan.md`。
> 3. 建立 `docs/PROGRESS.md`，表格列：任务号 / 状态 / 新增文件 / 新增测试 / 遗留问题。
> 4. `resihp/train.py` 目前只需能被 `python3 -m resihp.train --config ... --failures ...` 调起并打印解析后的参数，不要写任何训练逻辑。
> 5. 加一个 `tests/test_entrypoint.py`：验证唯一入口可解析参数、且仓库中不存在 `hello_dist.py`。
>
> 门禁：`python3 -m pytest -q` 全绿。完成后更新 PROGRESS.md 并停下。

---

## 阶段 1：Planner 纯函数（全部无 torch.distributed，可单进程测）

### T1 — 配置与故障文件校验

> 读 `CLAUDE.md`，再读计划文档「一、目标与约束」的故障文件部分与「四.A」。输出约束检查清单。
>
> 任务：实现 `resihp/config.py`。
> - 解析并校验 `train.json`（模型维度、层数、heads、batch、micro-batch、seed、初始 TP/PP/DP、迭代数）。
> - 解析并校验 `failures.json`：只允许 `after_iteration` 与 `failed_rank` 两个字段；`after_iteration` 严格递增；`failed_rank` 合法（在初始 world size 内）且不重复；**出现速度/降级/检测类字段直接拒绝并报出字段名**。
> - 校验失败抛结构化异常，含明确根因，不做任何默认值兜底。
>
> 先写 `tests/test_config.py`：合法样例、字段缺失、迭代非递增、rank 越界、rank 重复、含 `speed`/`p_i`/`detector` 字段被拒。再写最小实现。
>
> 门禁：`python3 -m pytest -q` 全绿。完成后更新 PROGRESS.md 并停下。

### T2 — 解析显存模型（唯一计算器）

> 读 `CLAUDE.md`，再读计划文档「3.3」的显存预算描述与「四.F 显存公式专项」。输出约束检查清单。
>
> 任务：实现 `resihp/memory.py`，提供**全项目唯一**的显存计算器：给定（模型配置、TP degree、该 stage 层数、micro-batch 数与形状），返回分片参数 + 复制参数 + 梯度 + AdamW 两份矩状态 + 峰值 activation 的字节数，并提供 `memory_feasible(...) -> bool`。
> TP 的 `k_min` 与 DP 重路由的 `MemoryFeasible` 必须调用同一函数，禁止出现第二份公式。
>
> 先写 `tests/test_memory.py`：至少两个**手算核对**的已知配置逐项比对字节数；再加边界用例（刚好满足 / 超一字节）。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

### T3 — TP 候选生成与确定性成员选择

> 读 `CLAUDE.md`，再读计划文档「3.3」。输出约束检查清单。
>
> 任务：实现 `resihp/planner/tp.py` 的纯函数：
> - `G' = G \ F_stop`；
> - 候选 degree `K = { k | k_min ≤ k ≤ |G'|, k = 2^q }`，且 attention heads 与 MLP 分片维度可被 k 整除，且显存可容纳（调 T2 计算器）；
> - **取最大可行 degree，成员按 rank 升序确定**。不要实现 `p_i` 评分分支，不要留多套 fallback。
> - 无可行候选时返回结构化不可行原因。
>
> 先写 `tests/test_planner_tp.py`：2 的幂候选生成；整除约束；`k_min` 显存约束；最大可行 degree 与确定性成员；单次掉卡；连续掉卡；无可行 TP。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

### T4 — PP 重分层纯函数

> 读 `CLAUDE.md`，再读计划文档「3.4」。输出约束检查清单。
>
> 任务：实现 `resihp/planner/pp.py`：
> - `new_tp_degrees` 是每个既有 stage 的新 TP degree，不是目标层数；`L_target = floor(L_old × TP_new / TP_old)`，有 TP 组的 stage 至少 1 层，TP 组完全失效的 stage 为 0 层；旧布局允许已有空 stage（旧层数与旧 TP degree 同为 0），以支持连续重分层；
> - 调整初始目标层数以守恒原模型总层数：正差值按当前「层数/TP degree」最小者、再按 stage ID 分配；负差值按该比值最大者、再按 stage ID 逆序收回，active stage 至少保留 1 层；比较使用整数交叉乘法；
> - 每 stage 为连续区间；所有全局 layer 恰出现一次且总层数守恒；embedding / LM head 归属首 / 尾可执行 stage；
> - 输出「哪些全局 layer 从哪个 stage 迁到哪个 stage」的迁移列表（本任务只算计划，不搬张量）。
>
> 先写 `tests/test_planner_pp.py`：论文示例 `(4,4,4)→(5,2,5)`；奇数余层与确定性余数；首/中/末 stage 故障；stage 清空；连续多次重分层；layer 连续唯一完整；embedding/LM head owner。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

### T5 — DP 确定性重路由纯函数

> 读 `CLAUDE.md`，再读计划文档「3.5」与「二、原则 B」。输出约束检查清单。
>
> 任务：实现 `resihp/planner/dp.py`：
> - 纯函数 `assign(step, failure_signature, active_topology) -> assignment`，同输入必同输出，**幂等**；
> - `failure_signature` 由「到当前事件为止已累计失效的 rank 集合」唯一确定；
> - 先按新拓扑可达容量做容量比例静态划分，再把失效 stage 的 micro-batch 按 replica ID 升序等确定规则重路由到健康 peer stage；
> - 目标可行性用 T2 的同一显存计算器判定；不可行按确定顺序试下一候选，全不可行返回结构化不可行；
> - 每个 micro-batch·每个 stage 恰执行一次。
> 禁止任何运行时延迟/进度输入。
>
> 先写 `tests/test_planner_dp.py`：容量划分与目标排序；失效 stage 全部 micro-batch 被迁出；执行完整性；显存允许/拒绝/边界；多目标平局确定性；同输入幂等。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

### T6 — ExecutionPlan 结构、版本与摘要

> 读 `CLAUDE.md`，再读计划文档「3.2」。输出约束检查清单。
>
> 任务：实现 `resihp/plan.py`：不可变、带版本号的 `ExecutionPlan`，**先只放最小必要字段**：活跃/失效 ranks；每个逻辑 DP replica 与 PP stage 的 TP 成员；每 stage 连续 layer 范围；每个 micro-batch·stage 的执行 rank；参数/优化器状态的 donor、目标与重分片方式；计划版本；规范化摘要（稳定哈希）；不可行时的结构化原因。
> 再实现 `build_plan(...)` 把 T3/T4/T5 串成一个计划，以及不变量断言函数 `assert_invariants(plan)`（计划文档四.B 全部条目）。
>
> 先写 `tests/test_plan.py`：同输入摘要稳定；版本严格递增；不变量断言能抓出人为构造的重叠 layer / 漏掉 micro-batch / rank 重复；不可行计划带根因。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

---

## 阶段 2：训练基线与 checkpoint

### T7 — 单进程确定性参考训练

> 读 `CLAUDE.md`，再读计划文档「3.1」。输出约束检查清单。
>
> 任务：实现 `resihp/model.py` 与 `resihp/reference.py`：自建 token/position embedding、LayerNorm、多头因果自注意力、MLP、残差、LM head；**每个 Transformer layer 用稳定全局 ID**，逻辑参数名在重规划前后保持稳定。固定模型初始化、输入 token、数据顺序、RNG 状态。单进程参考训练记录每轮输入、loss、参数摘要、优化器摘要。
>
> 先写 `tests/test_reference.py`：固定种子下输入/loss/一步更新可重复；参数命名与全局 layer ID 稳定。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

### T8 — checkpoint 原子读写 + 双口径断言骨架

> 读 `CLAUDE.md`，再读计划文档「3.1」「3.6」「二、原则 A」。输出约束检查清单。
>
> 任务：实现 `resihp/checkpoint.py`：保存完整逻辑模型、AdamW（`exp_avg`/`exp_avg_sq`/`step`）、迭代号、数据游标、CPU/CUDA RNG 状态、执行计划版本。**先写临时文件、完整校验后原子替换**；恢复成功前不覆盖最后有效 checkpoint；只保留当前恢复所需的唯一最新 checkpoint。
> 同时实现 `resihp/verify.py` 的两个断言：`assert_equals_checkpoint(state, ckpt)`（逐张量 `torch.equal` + dtype 一致）与 `assert_matches_reference(run, ref_run)`（逐步全精度一致）。参考基准的定义是「同一 checkpoint 起点 + 新拓扑 + 新配置实际 batch + 同一种子」，**不是无故障从头**。
>
> 先写 `tests/test_checkpoint.py`：save/load 后下一步与不中断训练一致；参数缺失/形状错/游标不一致明确失败；临时文件不完整时不替换旧 checkpoint；损坏 checkpoint 报根因。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

---

## 阶段 3：控制面与真实并行

### T9 — 控制面与安全点九步骨架

> 读 `CLAUDE.md`，再读计划文档「3.2」。输出约束检查清单。
>
> 任务：实现 `resihp/control.py` 与 `resihp/train.py` 主循环骨架：所有进程持有**始终存活的 Gloo 控制组**；训练组按当前计划建（GPU=NCCL / CPU=Gloo）。实现故障安全点的严格九步顺序：完成并提交当前迭代 → 原子保存 checkpoint → 广播 fail-stop 事件 → 标记 rank 永久失效 → TP→PP→DP 重规划 → 所有进程按统一顺序释放旧训练组并建新组 → 恢复/迁移/重切（本任务先留明确的调用点，不实现搬运）→ 校验计划摘要与状态摘要一致 → 从下一迭代继续。
> 本任务用 8 进程 Gloo 跑通「建组—毁组—重建」，训练可以还是最简单的前反向。
>
> 测试：`tests/test_control.py`（8 进程 Gloo，pytest 内起 torchrun 或 mp.spawn）：每事件只生成一个新计划版本；失效 rank 永久退出训练路径；各 rank 计划摘要一致；无死锁、无残留进程组。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

### T10 — 真实 TP 前反向（数值锁定）

> 读 `CLAUDE.md`，再读计划文档「3.3」的真实 TP 部分。输出约束检查清单。
>
> 任务：实现 `resihp/parallel/tp.py`：attention-head 与 Q/K/V 分片、MLP column-parallel、attention/MLP output row-parallel、必要 TP all-reduce、embedding 与 LM head 唯一固定分片。**本任务不做重切，只做静态 TP 的正确执行。**
>
> 测试：TP1 与 TP2 的前向、反向与一步 AdamW 更新，**与 T7 单进程参考数值精确一致**（`torch.equal`）。
>
> **承接 T9（本任务范围内接入）**：T9 的 safe-point 第 2 步（原子 checkpoint）与第 7 步（恢复/迁移/重切）在 `resihp/control.py` 中先留成空调用点 `ControlPlane._commit_checkpoint` / `_recover_state`——因 T9 骨架训练用 all-reduce，无真实逻辑状态可存搬。本任务引入真实 TP 训练状态后，把 T8 的 `save_checkpoint` 接进第 2 步；第 8 步 `agree_on_digest` 的 `state_digest` 由空串占位改为真实逻辑张量摘要。第 7 步真正的收集/重切随 T11–T14 落地，此处只做与真实状态对齐的最小接入。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

### T11 — TP 重切与异构 TP 边界

> 读 `CLAUDE.md`，再读计划文档「3.3」的重切与异构边界部分。输出约束检查清单。
>
> 任务：实现 degree/成员变化时的重切：① 从其他健康 DP replica 收集完整逻辑张量；② 某 shard 在所有健康 replica 均缺失时才从故障前 checkpoint 恢复；③ 按新 degree 重切 `param/grad/exp_avg/exp_avg_sq`；④ 分发新 TP 组；⑤ gather 后与 checkpoint 完整逻辑状态逐张量校验。
> 异构 TP 边界（只要功能正确，不做 P2P 性能优化）：前向 leader gather → 计算 → scatter/broadcast；反向对应 scatter-reduce，确保梯度不重复累加、不丢失。
>
> 测试：`TP2→TP1`；TP 成员替换的 gather/reshard 无丢失；**异构 TP 边界反向梯度与参考逐元素一致**；donor 恢复路径与 checkpoint fallback 路径分别单测。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

### T12 — PP 运行时与层迁移

> 读 `CLAUDE.md`，再读计划文档「3.4」。输出约束检查清单。
>
> 任务：实现 `resihp/parallel/pp.py`：由计划生成 Forward/Backward/Send/Recv/WeightUpdate 原语，统一 **1F1B 功能调度**（内部保留 F/B/W 三类）。实现层迁移：迁移 layer 一并迁移 `param/grad/exp_avg/exp_avg_sq/step` 与元数据；接收 stage 的 TP degree 不同则**直接按目标布局重切，不留旧布局兼容**。
>
> 测试：每层所有训练状态迁移前后逐张量一致；PP 两阶段前反向与一步更新与参考一致；TP 重切 + PP 移层组合后 shard/owner/通信边界一致。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

### T13 — DP 跨 replica 执行

> 读 `CLAUDE.md`，再读计划文档「3.5」的运行时保证部分。输出约束检查清单。
>
> 任务：把 T5 的 assignment 接到真实执行：forward activation 发给实际下游 executor、backward gradient 返回实际上游 executor；activation 在对应 backward 完成前持续计入显存；workload 不同时归属源与目标；executor 改变不影响 global batch 梯度归一化；不同 replica 的 PP 分层与 TP degree 可以不同。
>
> 测试：activation 生命周期；跨 replica 前向 activation 与反向 gradient；不同 PP 分层 / 不同 TP degree 的 replica 间迁移后**梯度与参考一致**。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

### T14 — 原子重配与一致停止

> 读 `CLAUDE.md`，再读计划文档「3.6」。输出约束检查清单。
>
> 任务：把恢复链路收敛成**唯一路径**：故障 → 读 checkpoint / 从健康 replica 收集 → 状态恢复 → 重排计划 → 建群 → 继续。缺失 shard 优先从健康 replica 收集，全缺失才走 checkpoint；恢复为完整逻辑张量后仅按新计划重新分片。
> 实现六条一致停止条件（无可行 TP degree / 无可执行 PP 分区 / DP 无满足显存目标 / checkpoint 缺失损坏摘要不匹配 / 各 rank 计划不一致 / 状态重切后完整逻辑张量不一致）：所有 rank 达成一致后保留故障前 checkpoint、输出结构化根因、正常退出，**不发布半完成计划、不允许部分 rank 继续或死锁**。
>
> 测试：每条停止条件一个用例，验证超时前全体一致退出、无半完成组/计划、最后 checkpoint 可重载、报错只指根因。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

---

## 阶段 4：组合与端到端

### T15 — 两两组合测试

> 读 `CLAUDE.md`，再读计划文档「四.D」。输出约束检查清单。
>
> 任务：补齐两两组合测试，**必须执行真实前反向，不只比计划**：TP+PP、TP+DP、PP+DP、Scheduler+建群、Scheduler+状态迁移、状态迁移+checkpoint、动态通信组+PipelineRuntime。只补测试与必要修复，不新增功能。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

### T16 — 完整 3D 端到端（8 进程 Gloo）

> 读 `CLAUDE.md`，再读计划文档「四.D」的完整 3D 场景。输出约束检查清单。
>
> 任务：实现端到端场景测试：8 进程 Gloo `TP2×PP2×DP2` — 无故障参考 → 第 2 轮后失效一个 TP rank → TP 重切 / PP 移层 / DP 重路由 → 恢复 → 第 4 轮后失效另一个 DP replica 的 rank → 再次重配恢复 → 训练至结束。
> 校验：无 collective 顺序错、无死锁；失效 rank 不再训练；每次故障仅一个新计划；**恢复前状态精确等于 checkpoint；恢复后与新配置参考一致**；数据不重不漏；layer 与 micro-batch·stage 不重不漏。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

### T17 — 随机故障序列与反复注入

> 读 `CLAUDE.md`，再读计划文档「四.C / 四.D / 四.E」。输出约束检查清单。
>
> 任务：
> 1. 固定 seed 的随机故障序列：随机安全点 + 随机 rank，逐次注入（每次单 rank）直到资源耗尽，全程校验 B 组不变量。
> 2. 反复注入（固定间隔坏一个）验证稳定性与无泄漏（进程组、显存、临时文件）。
> 3. 多故障点位覆盖：每 N step、每 2N step、随机间隔，并做多次序列回归。
> 4. 资源耗尽与错误注入场景（四.E）补全：TP 候选空；PP 无法覆盖全层；DP 目标显存全不足；健康 donor 全失但 checkpoint 可用；checkpoint 损坏；rank 间计划摘要不一致；状态迁移摘要错误。
>
> 门禁全绿后更新 PROGRESS.md 并停下。

### T18 — GPU/NCCL 验收与结项自检

> 读 `CLAUDE.md`，再读计划文档「五、交付顺序」与「六、假设与完成标准」。输出约束检查清单。
>
> 任务：
> 1. 用 `torchrun --standalone --nproc_per_node=8 -m resihp.train --config configs/train.json --failures configs/failures.json` 在 GPU/NCCL 下做验收，要求 **≥2 次连续 fail-stop 并继续训练**。
> 2. 全仓扫描并报告：代码中是否存在 `Detector`、`p_i`、速度/降速分支、`standby`、`Algorithm 1`、旧入口、前向兼容逻辑——**必须为零**，若有则删除。
> 3. 逐条对照「六、完成标准」出一份自检报告写入 `docs/ACCEPTANCE.md`：A–F 测试是否全通过、TP/PP/DP 是否真实执行、参数/梯度/AdamW/迭代号/数据游标是否无丢失、恢复前后双口径是否成立、资源不足时是否全体一致退出。
>
> 完成后更新 PROGRESS.md 并停下。

---

## 使用建议

- **每个任务开一个新会话**，上下文只留 `CLAUDE.md` + 计划文档 + 本任务相关文件。
- 任务失败或跑偏时：不要在同一会话里反复纠，直接 `/clear` 重开，把「上次偏在哪」写进指令的约束检查里。
- 若某任务代码量超过 ~400 行，说明拆得不够细，按测试用例再切一半。
- T3~T5 之间没有依赖，可以并行开三个会话；T9 之后必须严格串行。
