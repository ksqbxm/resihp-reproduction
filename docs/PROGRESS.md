# ResiHP 实现进度

> 唯一进度文档：后续只在本文件记录进度。

| 任务号 | 状态 | 新增文件 | 新增测试 | 遗留问题 |
|---|---|---|---|---|
| T0 | 实现完成，待审核 | `resihp/__init__.py`、`resihp/train.py`、`configs/train.json`、`configs/failures.json`、`docs/ResiHP_failstop_recovery_plan.md`、`docs/PROGRESS.md` | `tests/test_entrypoint.py` | 当前入口仅解析并打印配置，尚无训练逻辑；待审核后进入 T1 |

## T0 验证记录

- 已删除旧入口 `hello_dist.py`、旧测试脚本 `nccl_test.py`。
- 已执行 `python3 -m pytest -q`：2 passed。
- 已执行 `python3 -m resihp.train --config configs/train.json --failures configs/failures.json`：成功打印解析后的 JSON 参数。
- 已确认 `hello_dist.py` 与 `nccl_test.py` 均已删除。
- 之前关于“PyTorch 不支持 RTX 5090 的 `sm_120`、导致 `all_reduce` 卡死”的结论已更正：实际环境中 `sm_120` 计算正常，NCCL `all_reduce` 正常，2 卡测试已成功输出 `2.0`。此前失败的真实原因是目标 GPU 显存 OOM，原因是已有进程占用显存。
- 运行注意事项：执行多卡任务前先用 `nvidia-smi` 确认目标 GPU 空闲；或者通过 `CUDA_VISIBLE_DEVICES` 指定空闲 GPU。

## T2：解析显存模型

状态：实现完成，待审核。

新增：`resihp/memory.py`、`tests/test_memory.py`。

验证：
- `python3 -m pytest -q`：18 passed。
- 两组已知配置完成独立手算常量的逐项核对，测试期望值不再复用实现表达式。
- 按计划文档 3.3 对“embedding 与 LM head 唯一固定分片”的核实：该句要求它们使用固定的 TP shard 布局，固定的是归属/布局而不是每个 rank 持有整份。因此 `replicated_parameters` 按 `tp_degree` 分片；它们不作为完整复制参数重复计数。
- 按计划文档 3.5 的 activation 生命周期约束修正：峰值只计入尚未完成对应 backward 的 `in_flight_micro_batches`，不再把全部 `micro_batches` 线性相加。激活模型拆为两部分：attention/MLP 内部中间激活随 TP 分片并除以 `tp_degree`；TP all-reduce 后的 stage 输入/输出在每个 TP rank 都是完整份，不除 `tp_degree`。此前 activation 完全不含 `tp_degree` 等价于假定 TP=1，现已修正并在代码注释中写明物理依据。
- `memory_feasible` 已验证刚好达到预算时通过、低于预算 1 字节时拒绝。
- 已覆盖非法 TP degree、stage 层数和 micro-batch 数。
- 已修复 `resihp/memory.py` 中 `memory_feasible` 的缩进错误；修复前按实际仓库运行 `tests/test_memory.py` 会在收集阶段报 `IndentationError`。

后续审阅修正（语义统一为"分片"）：
- 字段 `MemoryBreakdown.replicated_parameters` → `boundary_parameters`：计算一直按 `tp_degree` 分片，旧名"复制"与算法/docstring 三处打架。改名后统一到分片语义，仍单列此项以便计划四.F 逐项手算核对边界权重字节。`total` 与 `tests/test_memory.py`（两处期望值）一并改名。
- 删除 docstring 里"名字说复制、其实是分片"的找补解释，改为正面陈述：embedding/LM head 与普通层一样按当前 TP degree 分片。
- 传导到 T3：`resihp/planner/tp.py` 的 k_min 显存过滤处补注释——含 `boundary_parameters` 在内的每一项都随 `degree` 分片、整体随 k 变化，无固定每-rank 常数开销，防止后续把边界项当常数。
- 补 activation 近似说明：边界激活只计一层、未随 `stage_layers` 展开，故 activation 是刻意的**下界估计**（docstring 与代码注释各一句）。

## T3：TP 候选生成与确定性成员选择

状态：实现完成，待审核。

新增：`resihp/planner/__init__.py`、`resihp/planner/tp.py`、`tests/test_planner_tp.py`。

验证：
- `python3 -m pytest -q tests/test_planner_tp.py`：12 passed。
- `python3 -m pytest -q`：31 passed。
- 已覆盖 2 的幂候选、attention heads / model dimension 整除、`k_min` 下界（含非 2 的幂时选择下一个满足 `k >= k_min` 的 2 的幂候选）、显存约束委托、最大可行 degree、rank 升序成员选择、单次/累计 stage-local 掉卡、无可行 TP 的结构化原因和确定性重复调用。
- 二次审核修正：`active_ranks` 的语义已在 `feasible_degrees` 文档中锁定为单个 stage / 物理通信域内的存活 rank 集合，即 `G' = stage_ranks - F_stop`；本函数只排序并从该集合中确定性选成员，不从全局 rank 推断 stage 归属。
- 二次审核修正：非法 `model_dim % num_heads != 0` 现在作为输入错误直接抛出，不再伪装为 `no_feasible_tp`；输入校验先于空候选返回执行，避免其他非法参数被早退短路。
- 二次审核修正：当传入 `memory_budget` 时，调用方必须显式传入 `sequence_length`、`vocab_size`、`in_flight_micro_batches`，防止默认 `1` 静默低估显存。显存判断仍统一调用 `resihp.memory.memory_feasible`，后续 1F1B 调用方必须传入 stage 实际峰值。
- 小修：将显存输入改为带字段名的 `_MemoryInputs(NamedTuple)`，消除四元组位置解包造成的静默错配隐患；单个 rank 的非负整数校验改为明确的单数错误信息。
- 格式修正：统一 `_feasible_degrees` 中显存过滤分支的缩进，避免格式检查报 `E117 over-indented`；逻辑保持不变。

## T4：PP 重分层纯函数

状态：实现完成，待审核。

新增：`resihp/planner/pp.py`、`tests/test_planner_pp.py`。

本轮修正：
- 锁定 `new_tp_degrees` 为新 TP degree，测试不再把目标层数当输入；允许连续重分层传入零层/零 TP 的旧空 stage。
- 差值调整使用整数交叉乘法；明确正差分配与负差收回规则；返回前验证 active stage 可执行性与总层数守恒。
- 测试独立校验公式、差值分支、空 stage 连续调用、区间/迁移一致性、输入边界与大整数比例。
- 已验证 `python3 -m pytest -q tests/test_planner_pp.py`：16 passed；全量 `python3 -m pytest -q`：47 passed。

## T5：DP 确定性重路由纯函数

状态：实现完成，待审核。

新增：`resihp/planner/dp.py`、`tests/test_planner_dp.py`。

验证：
- `python3 -m pytest -q tests/test_planner_dp.py`：9 passed。
- 全量 `python3 -m pytest -q`：56 passed。
- 已覆盖容量比例静态划分、按 replica ID 确定性平局、失效 TP 成员导致 stage 迁出、每个 micro-batch·stage 恰执行一次、显存边界/拒绝、失败签名归一化与幂等。
- DP 显存判断统一调用 `resihp.memory.memory_feasible`，不引入第二份公式；赋值结果不依赖运行时延迟或进度。

## T6：ExecutionPlan 结构、版本与摘要

状态：二次修复完成，待审核。

新增：`resihp/plan.py`、`tests/test_plan.py`；本轮同时最小改动 `resihp/planner/dp.py`（支持异构活跃 stage）与 `tests/test_planner_dp.py`（新增异构用例）。

验证：
- 全量 `python3 -m pytest -q`：74 passed（`tests/test_plan.py` 17；`tests/test_planner_dp.py` 10）。
- `build_plan(config, *, step, version, failed_ranks=(), previous=None, ...)`：`step`（训练迭代）与 `version`（计划版本）分离，`assign(step, ...)` 只收 step；返回前自动执行 `assert_invariants(plan, previous)`。
- **单 stage 全失效不删整 replica**：该 stage 交给 T4 作 `tp_degree=0`/空成员重分层，其 layer 迁到本 replica 存活 stage，健康 rank 不被丢弃；仅当某 replica 全部 rank 失效才 skip，全部 replica 皆亡才 `no_surviving_replica`。
- **连续故障基于 previous plan**：迁移以上一个计划的 TP degree/成员与 layer 区间为旧布局，migration/state route 表示 `previous → current`，不再从初始 config 重算，避免第二次故障产生错误 source。
- **显存归口 TP 阶段**：`choose_tp` 传入 `memory_budget`、stage 层数、seq/vocab、micro/in-flight，显存不可行直接以 `no_feasible_tp` 报出；DP 层不再做显存兜底（`assign` 只做容量比例划分），消除「先选放不下的 degree 再靠 DP no_feasible_dp_target」路径。
- **拓扑不按 placements 删除**：`stages` = 当前可用执行拓扑（全部存活 replica），`placements` = workload；本轮分到 0 个 micro-batch 的健康 replica 仍保留在拓扑中。
- **状态迁移计划**：`StateRoute{replica, layer, target_ranks, donor_kind(prev_owner/peer_replica/checkpoint), donor_ranks, reshard(peer_copy/gather_reshard/checkpoint_restore), states(param/grad/exp_avg/exp_avg_sq/step)}`；donor 优先本 replica 旧属主（全存活）→ 健康 peer replica → checkpoint。仅生成计划，不搬张量。
- **消除重复状态**：`StagePlan` 只存 `layer_range`，`stage_layers` 改为只读属性；不再保留独立 `stage_layers` 字段与 `LayerMigration`。
- **摘要覆盖完整计划**：`digest` 用排序后 JSON 规范序列化（非 repr），覆盖 version/step/影响拓扑的 config 字段/active·failed ranks/stages/placements/state_routes；任一语义变化都改变摘要。`active_ranks` 单一语义=已分配到活跃 stage 的 rank；`live_ranks`（=world−failed）单列属性。
- `assert_invariants(plan, previous=None)`：`tp_degree==len(members)`；rank 不重复、不含 failed、active==assigned；每 replica 活跃 stage 的 layer 连续/唯一/无重叠/无缺口且守恒；每 micro-batch 恰在一个 replica、恰跑该 replica 全部活跃 stage 一次（异构合法）；placement executor 与 stage 成员一致；state route donor/target/reshard 合法；传入 previous 时版本严格递增、active 单调收缩、failed 单调增长。
- DP `assign` 最小改动：去掉「所有 replica 必须拥有全部 stage_id」的统一性要求，每个 replica 按自身活跃 stage 组流水线；容量比例分配算法未改，既有 uniform 用例行为不变。

### 本轮修复的原问题根因
1. 用 `if any(not members): continue` 在 planner 前就按单 stage 失效删除整个 replica，顺带丢弃同 replica 健康 rank——应交 T4/T5 判定。
2. 把计划 `version` 当作 `assign` 的训练 `step` 传入，混淆两个语义。
3. 迁移每次都从初始 config 与初始 layer 分布计算，连续故障时 source stage 错误。
4. 只有 `LayerMigration`（层的 stage 变化），缺 donor/target/reshard 的状态路由。
5. TP 显存约束未在 TP 阶段生效，靠 DP 返回 `no_feasible_dp_target` 兜底。
6. 用 `placed`（有 placements 的 replica）反推并裁剪 `stages`，令 0 workload 的健康拓扑消失。
7. `stage_layers` 与 `layer_range` 两份可能不一致的重复状态；`digest` 用 `repr` 且不含 version/step。

### 仍超出 T6 范围的遗留
- TP 显存判定在重分层之前进行，用旧 stage 层数估计；重分层后层数变大的显存复检属 T11 真实重切范畴，本任务不做。
- `StateRoute` 只描述迁移计划（donor/target/reshard/states），真实张量收集、gather+reshard、checkpoint 读取在 T11–T14 实现。
- 存在暂时未分配的健康 rank（仅当 TP>2 且降级留空位）时的「重新启用」策略未实现；锁定配置 TP=2 下 active==live，不触发。

## T7：单进程确定性参考训练

状态：实现完成，**门禁待 torch 环境执行**（本机无 torch，计划硬性禁止安装/升级 torch，故 T7 可执行门禁需在带 torch 的目标机跑）。

新增：`resihp/model.py`、`resihp/reference.py`、`tests/test_reference.py`。

设计要点：
- **自建 decoder-only Transformer**：token/position embedding、LayerNorm、多头因果自注意力（Q/K/V/O 分片投影 + 因果 mask + softmax）、MLP（fc1→gelu→fc2，隐层 4×）、pre-norm 残差、final LayerNorm、独立 LM head。用 `nn.Linear/nn.Embedding/nn.LayerNorm` 作原语，架构自行组装，不用现成 Transformer。
- **稳定全局 layer ID**：Transformer block 放进以字符串 gid 为键的 `nn.ModuleDict`（`self.layers`），逻辑参数名固定为 `layers.<gid>.attn.q_proj.weight` 等，重规划前后（PP owner / TP degree 变化）名字不随模块位置漂移。`layer_ids` 属性**从 `self.layers` 的键派生并升序**（键是唯一真源），`forward` 与所有摘要都遍历它、绝不用 `range(num_layers)`，从而遍历/持有与全局 ID 真正解耦，为后续 PP 迁移（stage 只持有非连续子集）留正确接口。
- **与唯一显存计算器对齐**：attention/MLP 线性层无 bias，使单层参数恰为 `12·model_dim² + 4·model_dim`，与 `resihp/memory.py` 的逐层字节数口径一致（4 个 attention 投影 = `4·d²`，2 个 MLP 矩阵 `d↔4d` = `8·d²`，2 个 LayerNorm 的 weight/bias = `4·d`）；测试 `test_layer_parameter_count_matches_memory_model` 独立核对。
- **确定性**：`run_reference` 内 `torch.manual_seed(config.seed)` 后按固定构造顺序建模（固定初始化），数据用独立 `torch.Generator` 生成固定 token 流（固定输入与数据顺序）；AdamW 为唯一优化器，next-token cross-entropy 为唯一损失，全 FP32、CPU 单进程。
- **每轮记录**：`StepRecord{step, tokens（输入）, loss, param_digest（参数摘要）, optim_digest（优化器摘要）}`；摘要用 sha256 逐张量精确 FP32 字节哈希（`name + shape + numpy().tobytes()`；param 覆盖全部逻辑参数，optim 覆盖每参数 `exp_avg/exp_avg_sq/step`）。`logical_state_dict()` 返回活引用而非拷贝（`test_logical_state_dict_returns_live_references` 锁定），供 T8 checkpoint / T11 reshard 直接消费。
- `vocab_size`、`sequence_length` 按现有 `memory.py`/`plan.py` 惯例作显式关键字参数传入，不改 T1 的 `TrainConfig`（避免越界改动）。

审阅后修正（本轮）：
- 严重：`layer_ids` 原写死为 `tuple(range(num_layers))`，"稳定全局 ID" 只在命名上成立、遍历/持有仍假设持有全部连续层；改为从 `nn.ModuleDict` 键派生升序，真源唯一。
- 删除 `model.py` 里只校验 `vocab_size`/`sequence_length` 的半吊子 `_positive`：config 三字段已由 T1 `config.py` 统一校验，`sequence_length≥2` 由 `run_reference` 保证，不再制造"已校验"错觉。
- 删除 `run_reference` 未被任何调用点/测试驱动的 `steps` 覆盖参数，直接用 `config.iterations`，不留多余灵活性。
- `_digest` 由 `struct.pack` 逐元素展开改为 `numpy().tobytes()`，等价且更短更快。
- 修正 model.py docstring 参数计数措辞（原文易被读成 MLP 为两个 `d²` 矩阵）。

本机可做的验证（无 torch）：
- `py_compile resihp/model.py resihp/reference.py tests/test_reference.py`：通过。
- 用 torch stub `import resihp.model, resihp.reference`：通过（无 import 期错误）。
- 全量 `python -m pytest -q`：74 passed, 1 skipped——`tests/test_reference.py` 顶部 `pytest.importorskip("torch")` 在无 torch 时整体跳过，不误伤门禁；既有 74 项无回归。

待目标机（带 torch）执行的门禁：
- `python3 -m pytest -q tests/test_reference.py`，覆盖：固定种子下输入/loss/一步更新可重复（两次 run 逐条 `==`）；数据顺序固定且各步 batch 相异；参数命名与全局 layer ID 稳定；`logical_state_dict` 返回活引用；单层参数数与显存模型一致；loss 有限；序列长度 <2 被拒。
- 通过后本节状态改为「实现完成，待审核」。

遗留问题：
- 参考训练**仅锁 CPU 确定性**（`torch.manual_seed` 只设 CPU RNG，未涉及 CUDA/`use_deterministic_algorithms`）。CPU/CUDA RNG 状态保存属 T8 checkpoint，GPU 确定性与"GPU 执行 vs CPU 参考 `torch.equal`"的可复现前提延后到 T10/T18。
- AdamW 超参（`LEARNING_RATE/WEIGHT_DECAY/ADAM_BETAS/ADAM_EPS`）目前硬编码在 `reference.py`。T8 要保存优化器状态、T10 要求数值一致，届时应收敛为单一真源（可能并入 config），避免两处不同步。
