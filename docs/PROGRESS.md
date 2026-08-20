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

## T8：原子 checkpoint 保存 / 恢复

状态：实现完成，**门禁待 torch 环境执行**（本机无 torch，计划硬性禁止安装/升级 torch，故 T8 行为门禁需在带 torch 的目标机跑）。

新增：`resihp/checkpoint.py`、`tests/test_checkpoint.py`；最小改动 `resihp/reference.py`（抽出可续训的 `ReferenceRun`）。

设计要点：
- **`reference.py` 抽出 `ReferenceRun`（外部行为不变）**：把原 `run_reference` 的建模/建优化器/生成 token 流与单步训练逻辑收进类，`step()` 前进一次迭代、`cursor` = 已完成步数（同时也是数据游标）。`run_reference` 改为「建 `ReferenceRun` → 循环 `step()`」的薄封装，逐算子顺序不变，`StepRecord`/两份摘要仍逐字节一致，既有 `tests/test_reference.py` 不受影响。续训是 T8 硬需求，故此重构属本任务范畴，非投机。
- **checkpoint 覆盖计划 3.1 全集**：完整逻辑参数、AdamW（`exp_avg`/`exp_avg_sq`/`step`）、`completed_steps`（=迭代号=数据游标，token 流是 `seed+index` 纯函数，单一游标即可确定下一 batch，不另存冗余游标）、CPU/CUDA RNG 状态、执行计划版本 `plan_version`（存 int，不耦合 T6 `ExecutionPlan` 对象）。另存 `identity`（只含决定逻辑参数形状与 token 数据流的字段：`model_dim/num_layers/num_heads/vocab_size/sequence_length/batch_size/seed`，**不含拓扑字段**，详见下方审阅修正）以锁定张量形状与 token 流来源。
- **原子写（计划 3.6）**：`save_checkpoint` 先写 `path.tmp` → 重新 `torch.load` 并按 sha256 逐张量字节摘要与内存态核对（写入校验）→ `os.replace` 原子替换到唯一 canonical 路径；任何异常都 `unlink` 临时文件并向上抛，最后有效 checkpoint 不被覆盖，磁盘上恒为单一文件。
- **加载即校验，缺一即明确失败**：`load_checkpoint` 依次校验格式版本、`identity` 一致、参数不缺/不多且逐个形状匹配、`completed_steps` 与每个参数的 AdamW `step` 一致（游标一致性）；任一失败抛 `CheckpointError` 并指明根因。恢复顺序：`copy_` 回参数 → 清空并按 param 对象重建 optimizer.state → 恢复 RNG → 置 `cursor`。

本机可做的验证（无 torch）：
- `py_compile resihp/reference.py resihp/checkpoint.py tests/test_checkpoint.py`：通过。
- 用 torch stub `import resihp.model, resihp.reference, resihp.checkpoint`：通过（无 import 期错误）。
- 全量 `python -m pytest -q`：74 passed, 2 skipped——`tests/test_reference.py` 与 `tests/test_checkpoint.py` 顶部 `pytest.importorskip("torch")` 在无 torch 时整体跳过；既有 74 项无回归（`run_reference` 重构未改变外部行为）。

审阅后修正（本轮）：
- **正确性（根本）**：`_identity` 原用 `asdict(config)` 把 `tp/pp/dp/micro_batch_size` 也焊进 checkpoint 身份，与恢复设计（原则 A：同一 checkpoint 起点 + 新拓扑）冲突——T10/T11「读 checkpoint → 换拓扑重切」会在 load 身份校验处被误拒。改为只保留决定「逻辑参数形状 + token 数据流」的字段（`model_dim/num_layers/num_heads/vocab_size/sequence_length/batch_size/seed`）；拓扑由 `plan_version` 追踪，不作 config 相等性把关。新增 `test_identity_ignores_topology_fields` 锁定「同逻辑/数据、异 TP/PP/DP」可正常续训。
- **完备性**：`_validate` 的游标一致性检查原为「遍历 optim 逐个比 step」，当 `completed>0` 而 `optim` 被篡改成空/缺项时循环空转、静默放行，恢复出残缺 AdamW 状态。补「`completed>0` 时 `set(optim)` 必须恰等于参数集合」，missing/extra 均明确报错。新增 `test_missing_optimizer_state_is_rejected`。
- 小修：删去因 `_identity` 改写而多余的 `from dataclasses import asdict` 导入；`test_config_mismatch` 改用 `dataclasses.replace` 构造异 seed 配置，替代 `CONFIG.__dict__`。

待目标机（带 torch）执行的门禁：`python3 -m pytest -q tests/test_checkpoint.py`，覆盖：save→load 后续训的下一步与不中断训练逐字段一致；成功保存后磁盘仅剩唯一 canonical 文件、无残留 `.tmp`；`os.replace` 失败时原 checkpoint 完好且无残留临时文件；参数缺失 / 形状错 / 游标不一致 / 优化器状态缺失 / 配置不一致分别以 `CheckpointError` 明确拒绝；异拓扑同逻辑可续训；负 `plan_version` 被拒。通过后本节状态改为「实现完成，待审核」。

遗留问题：
- CUDA RNG 已保存/恢复，但仅在 GPU 上生效；CPU 门禁只覆盖 CPU RNG。GPU 确定性前提仍随计划延后到 T10/T18。
- `plan_version` 现为 int；与 T6 `ExecutionPlan` 版本/摘要的绑定（故障恢复链路里按计划版本校验）留到接入分布式执行时再做，本任务不引入耦合。
- checkpoint 只服务单进程 `ReferenceRun`；「从健康 replica 收集 / 缺失 shard 才回落 checkpoint」的分布式恢复链路（计划 3.3/3.6）属 T11+。

## T9：控制面与安全点九步骨架

状态：实现完成，**分布式门禁待 torch 环境执行**（本机无 torch，计划硬性禁止安装/升级 torch，故 8 进程 Gloo 门禁需在带 torch 的目标机跑）。

新增：`resihp/control.py`、`tests/test_control.py`；`resihp/train.py` 加分布式主循环分支（非 torchrun 启动时保留原「解析并打印 JSON」行为，`tests/test_entrypoint.py` 不受影响）。

设计要点：
- **两个进程组，模拟式 fail-stop**：`ControlPlane` 持有**始终存活的 Gloo 控制组**（`WORLD`，全程不销毁，承载 fail-stop 广播与计划摘要一致性校验）；**训练组**是按当前 `ExecutionPlan.live_ranks` 建的子组（GPU=NCCL / CPU=Gloo），每次故障销毁重建。本复现用「排除」模拟 fail-stop：进程不真死，只被移出训练组、停做训练，仍留在控制组，从而让确定性故障表可测且控制组集合不死锁。
- **严格九步安全点**（`ControlPlane.safe_point`）：① 完成并提交当前迭代（调用方在进入前完成）→ ② 原子 checkpoint（`_commit_checkpoint` 调用点）→ ③ 控制组广播 fail-stop（src=0 协调者，本复现故障表不打 rank 0）→ ④ 标记 rank 永久失效（并入 `failed`）→ ⑤ TP→PP→DP 重规划（`reconfigure` 纯函数，版本 +1，`previous` 传旧计划）→ ⑥ 统一顺序先销旧训练组再建新组 → ⑦ 恢复/迁移/重切（`_recover_state` 调用点，本任务不搬张量，留给 T11–T14）→ ⑧ `all_gather_object` 校验各 rank 计划/状态摘要一致 → ⑨ 调用方从下一迭代继续。
- **通信 rank 只由当前计划决定**：训练组成员 = `plan.live_ranks`，`is_training_rank` 与建组同源，不从旧布局隐式推导。
- **非成员正确处理**：`new_group` 是 `WORLD` 级集合调用，全体（含失效 rank）都要进入；非成员拿到 sentinel，故 `build_training_group` 对非成员存 `None`，`destroy_training_group` 只销真实组——非成员既不跑训练集合，也不销毁自己没加入的组。
- **训练用最简 all-reduce** 代替真前反向（T9 只锁控制面「建组—毁组—重建」骨架）；真实 TP/PP/DP 执行属 T10+。

本机可做的验证（无 torch）：
- `py_compile resihp/control.py resihp/train.py tests/test_control.py`：通过。
- 全量 `python -m pytest -q`：75 passed, 3 skipped——新增纯函数用例 `test_reconfigure_is_deterministic_and_increments_version`（不依赖 torch，本机通过）验证重规划确定性与版本严格递增；`tests/test_control.py` 的 8 进程 Gloo 用例经 `skipif(torch 缺失)` 跳过；既有用例无回归。
- `python -m resihp.train --config ... --failures ...`（非 torchrun）：仍打印解析后的 JSON，`test_entrypoint.py` 行为不变。
- **离线仿真校验**（scratchpad 假 `torch.distributed`，逐 rank 记录集合调用序列）确认：8 个 rank 的 WORLD 级集合调用（`init`/`new_group`/`broadcast`/`all_gather_object`/`barrier`/销毁默认组）顺序完全一致（控制组不死锁）；每个训练组恰由其成员集合销毁（无残留、收支平衡：失效 rank 1 只销 1 组、rank 5 销 2 组、健康 rank 各销 3 组）；每次 all-reduce 的组都含调用者；失效 rank 在故障迭代后退出训练路径（rank 1 训 `[1,2]`、rank 5 训 `[1,2,3,4]`、其余训满 6 步）；结束后无 rank 残留 `is_initialized`。

待目标机（带 torch）执行的门禁：`python3 -m pytest -q tests/test_control.py`（8 进程 Gloo，`mp.spawn`），覆盖：每个 fail-stop 事件恰生成一个严格递增的新计划版本（各 rank `versions==[1,2]`）；失效 rank 永久退出训练路径（`trained` 分别为 `[1,2]`/`[1,2,3,4]`，健康 rank 满 6）；各 rank 每个事件的计划摘要一致；无死锁、结束后无残留进程组（`still_initialized==False`）。通过后本节状态改为「实现完成，待审核」。

遗留问题：
- ② checkpoint 与 ⑦ 恢复/迁移/重切仅为调用点：本任务训练无真实逻辑状态可存搬，`save_checkpoint`（T8）与 `state_routes` 驱动的收集/重切（T11–T14）在真实执行接入后再落地，届时 ⑧ 的「状态摘要」补入真实逻辑张量摘要（现为空串占位，各 rank 恒等）。
- src=0 广播依赖协调者 rank 0 存活；本复现确定性故障表不打 rank 0。若后续放开「任意 rank 可失效」，广播源需改为当前最低存活 rank，属 T14 一致停止/恢复链路范畴。
- 训练组 = `live_ranks`；锁定配置 TP=2 下 `active==live`。「健康但未分配」rank（TP>2 降级留空位）不参与训练却仍在训练组的取舍，与 T6 同一遗留，接入真实执行时再定。

## T10：真实 TP 前反向（真正分片 all-reduce，`allclose` 容差）

状态：实现完成，**分布式门禁待 torch 环境执行**（本机无 torch，计划硬性禁止安装/升级 torch，故 2 进程 Gloo TP 门禁需在带 torch 的目标机跑）。

新增：`resihp/parallel/__init__.py`、`resihp/parallel/tp.py`、`tests/test_parallel_tp.py`；最小改动 `resihp/reference.py`（新增 `logical_state_digest`）与 `resihp/control.py`（安全点 ②/⑧ 接入真实状态）。

### 验收口径调整（应要求：`torch.equal` → `torch.allclose`）
- **原则 A 的比特级 `torch.equal` 与真正的分片 all-reduce 不可兼得**：真正的分片计算引入两处「重排 FP32 求和」的归约——① 前向 row-parallel all-reduce（`out_proj`/`fc2`，`partial₀+partial₁`）；② 反向 column-parallel 输入梯度 all-reduce（Q/K/V/`fc1`）。浮点加法不满足结合律，分片部分和 ≠ 参考单块 matmul。
- 按要求**放宽为 `torch.allclose`**（允许微小 FP 误差），从而实现**多卡真正的分片 all-reduce**。TP degree=1 时所有 collective 退化为 no-op，仍与参考逐比特一致；degree=2 在容差内一致。
- **容差取 `rtol=1e-4, atol=1e-5`**（非默认 `atol=1e-8`）：q/k/v 与逐 head attention 是 column-parallel、无归约、与参考逐比特一致，误差**仅**来自 `out_proj`/`fc2` 的 2 路 all-reduce（单次 ~1e-6，几层累积后更大）；默认 `atol=1e-8` 对近零 logits 过严会误报。此容差反映真实 reassociation 误差，真实 bug 量级远大于此仍会失败。测试同时记录 `max_forward/loss/grad/param_diff` 以便区分「重排误差」与「真实 bug」。

### 真实 TP 执行（Megatron 两算子）
- 每个进程即一个 TP rank，**只持有自身 shard**（真正省显存，非"存整份"）：Q/K/V/`fc1` 按输出(head)维分片（column-parallel），`out_proj`/`fc2` 按输入维分片（row-parallel），token embedding / LM head 按 vocab 分片；两个 LayerNorm 与 position embedding 复制。布局与 `resihp/memory.py` 预算一致；每个 shard 是独立 leaf 参数、独立梯度、独立 AdamW 矩状态（正是 T11+ 收集/重切的对象）。
- `f`（`_CopyToRegion`）：前向恒等、**反向 all-reduce**，包住每个 column-parallel 区（Q/K/V、`fc1`）的输入，使复制的输入梯度在组内求和。
- `g`（`_ReduceFromRegion`）：**前向 all-reduce**、反向恒等，把每个 row-parallel 区（`out_proj`、`fc2`）与 vocab-parallel token embedding 的部分输出归约为完整复制结果。
- LM head column-parallel over vocab：各 rank 的分片 logits 经 **all-gather**（`_GatherLastDim`，反向取本地切片）拼成完整 logits，loss 用普通 cross-entropy。
- 因每处 `g` all-reduce 与残差加都在各 rank 产出相同字节，隐藏态**始终一致复制**，故复制的 LayerNorm/position-embedding 梯度各 rank 天然一致、无需额外通信即保持同步。
- TP 组用 `group`（默认 world）参数化，degree/rank 从组导出，可服务整组 TP 或 3D 布局中的 TP 子组。
- **不做重切**：degree/成员变化的 gather+reshard 留给 T11+；此处只做静态 TP 的正确执行。

### 承接 T9 的最小接入
- `ControlPlane` 新增可选 `training_run`/`checkpoint_path`（默认 `None`）。未附真实状态时安全点行为与 T9 骨架完全一致（② 空操作、⑧ state_digest 为空串），故 `tests/test_control.py` 不受影响、无回归。
- 附上真实状态后：② `_commit_checkpoint` 调 T8 `save_checkpoint`；⑧ `_state_digest` 由新增 `reference.logical_state_digest(model)` 产出完整逻辑参数摘要，`safe_point` 把它传给 `agree_on_digest`。⑦ 收集/重切仍为调用点（T11–T14）。
- 分片 TP run 的「完整逻辑 checkpoint」需先 gather optimizer 矩状态（属 T11 重切），本任务不做；wiring 单测用 `ReferenceRun`（完整逻辑）验证 ②/⑧ 机制打通。

本机可做的验证（无 torch）：
- `python -m py_compile resihp/parallel/tp.py resihp/reference.py resihp/control.py tests/test_parallel_tp.py`：通过。
- 全量 `python -m pytest -q`：75 passed, 4 skipped——`tests/test_parallel_tp.py` 顶部 `pytest.importorskip("torch")` 在无 torch 时整体跳过；`resihp/control.py` 仍为 import 期 torch-free（`save_checkpoint`/`logical_state_digest` 均方法内惰性导入），既有 75 项无回归。

待目标机（带 torch）执行的门禁：`python3 -m pytest -q tests/test_parallel_tp.py`，两条数值门禁跑同一 `_compare` 比对逻辑：
- CPU/**Gloo**（`test_tp_matches_reference_within_tolerance`，TP1/TP2 各 `mp.spawn` 起 1/2 进程）：始终可跑。
- GPU/**NCCL**（`test_tp_cuda_nccl_matches_reference`，TP1/TP2）：`torch.cuda.set_device(rank)` + `cuda:rank` + `nccl`，模型/输入/参考/shard 全上 CUDA，显式断言 `get_backend()=="nccl"`、`current_device()==rank`、`tp_logits.is_cuda`；CUDA 缺失或 GPU 少于 degree 才 `skip`，2-GPU 服务器上实跑。
两者覆盖：gather 后完整 logits 与 loss 与参考 `allclose`；各 rank 本地 shard 的**梯度**与一步 AdamW 后的**参数**与参考对应切片 `allclose`（degree=1 精确一致，degree=2 容差内）；均记录 `max_forward/loss/grad/param_diff`。另有不整除 degree 被拒、`ControlPlane` 附真实状态后 ② 落盘且可重载、⑧ 摘要为真实逻辑摘要（未附状态时为空串）。通过后本节状态改为「实现完成，待审核」。

遗留问题：
- 数值验收为 `allclose`（`rtol=1e-4, atol=1e-5`），非逐比特；两两组合/端到端（T15/T16）沿用同一口径与参考对照。
- 分布式后端：CPU 用 Gloo、GPU 用 NCCL 均已成门禁；8 进程完整 3D/NCCL 大规模验收仍属 T16/T18。
- TP 仅静态执行，无 degree/成员重切；gather+reshard 属 T11。
- `ControlPlane._commit_checkpoint` 目前只能存**完整逻辑** run；分片 TP run 的 checkpoint 需 T11 的 optimizer 矩状态 gather。⑦ `_recover_state` 仍空。

## T11：TP 重切与异构 TP 边界

状态：实现完成 + 一轮复审修正，**分布式门禁（Gloo + 真实 GPU/NCCL）待 torch 环境执行**（本机无 torch/numpy，计划硬性禁止安装/升级 torch，故 `mp.spawn` 门禁需在带 torch 的目标机跑）。

新增：`resihp/parallel/reshard.py`、`tests/test_parallel_reshard.py`。未改动 `tp.py`/`control.py`/`model.py`/`reference.py`（保持 surgical）。

设计要点：
- **单一恢复路径的纯函数化**：`reconstruct_full(name, shard_dim, old_size, contributions, checkpoint)` 是重构完整逻辑张量的唯一入口——① 全部 shard 索引在健康 rank 中齐备（同一逻辑层的 shard 也存在于健康 peer DP replica）→ 从 peer 拼接（`source="peer"`）；② 某 shard 索引在所有健康 rank 均缺失→回落故障前 checkpoint（`source="checkpoint"`）；③ 两处皆无→抛 `ReshardError`（一致停止条件，链路接入属 T14）。此函数不依赖任何进程组，故 **donor 恢复路径与 checkpoint fallback 路径分别以纯单测锁定**（计划 3.3 要求「分别单测」）。
- **分布式驱动是薄封装**：`reshard_tp_state(...)` 用一次 `all_gather_object` 汇总各健康 rank 持有的 `{name: {shard_index, param, grad, exp_avg, exp_avg_sq, step}}`，`_merge` 折叠成 `{name: {field: {shard_index: tensor}}}`，每个 rank 用同一 merged 视图独立重构 → 与 checkpoint **逐张量 `torch.equal` 校验**（计划步骤⑤）→ 按新 layout（`local_slice`）切出**本 rank 的新 shard**；被踢 rank（`new_rank=None`）仍参与 gather 供 peer 使用但不接收（返回 `{}`）。param/grad/exp_avg/exp_avg_sq 按参数 shard 维重切，`step` 作复制标量。
- **shard 布局单一真源**：`shard_dims(layer_ids)` 给出全模型 `logical_name → shard_dim|None`，与 `TensorParallelTransformer.local_shards()` 的分片维一致；`test_shard_dims_matches_tp_module` 用 tp=1 组实例化 TP 模块逐项核对二者相等，锁死漂移。
- **异构 TP 边界（功能正确，不做 P2P 优化）**：`cross_tp_boundary` = 自定义 autograd `_ReplicatedBridge`：前向从 `upstream_leader` 广播权威副本给边界组全体（下游 TP 组据此拿到激活）；反向从 `downstream_leader` 广播权威梯度回全体——因两侧在各自 TP 组内激活/梯度均复制，**只搬一份、绝不求和**，故上游每个 rank 拿到的边界梯度恰等于单进程参考（naive all-reduce 会按下游 degree 倍增而被测试抓出）。测试用**上游 TP1→下游 TP2**（下游 degree>上游，双 rank 均跑真实下游 loss，无需零缩放占位），既覆盖「不重复计数」又能在 2-GPU 机实跑 NCCL `broadcast`。

### 审阅修正（本轮，新增真实 GPU/NCCL 测试并复审）

- **正确性（GPU 致命，根因修复）**：`all_gather_object` 汇总的是各 rank 持有的**活 CUDA shard**，pickle 会给每个张量打上**属主的 device 序号**；`reconstruct_full` 里 `torch.cat([cuda:0 shard, cuda:1 shard])` 会**跨设备崩溃**。既有 Gloo 测试全在 CPU，永远碰不到这条路径。修复：新增 `_to_cpu`，在 `all_gather_object` 之前把本 rank shard 规范化为 CPU——完整逻辑张量本就是**设备无关的规范形态**（checkpoint 锚点也是 CPU），重构后返回 CPU shard，由调用方放回计算设备。这是根因修复而非兜底：跨设备 gather-then-cat 本身语义就错。
- **完备性（潜在 KeyError）**：原 `ckpt = {name: checkpoint[name][field]}` 对每个 field 都急切索引 checkpoint；真实 T8 checkpoint **不含 `grad`**，于是当 grad 由 peer 提供时也会 `KeyError`。修复：仅当 checkpoint 确实带该 field 时才取用（`ckpt_entry.get(field)`），peer 提供的 grad 正常重切。
- **离线逻辑校验**（scratchpad 假 torch/dist，纯 Python 张量执行 `reshard_tp_state` 真身）：degrade / replace 在 `cpu` 与 `cuda:0` 两种 device 下均通过；假 `torch.cat` 内置**单设备断言**，故若无 `_to_cpu` 则 `cuda:0` 用例会像真 torch 一样崩——通过即证明修复生效而非掩盖。另验证 checkpoint fallback 与「grad 不在 checkpoint」不再 KeyError。

本机可做的验证（无 torch）：
- `python -m py_compile`（全包 + 测试）：通过。
- 全量 `python -m pytest -q`：75 passed, 5 skipped——`tests/test_parallel_reshard.py` 顶部 `pytest.importorskip("torch")` 在无 torch 时整体跳过（较 T10 基线 4 skipped 增 1）；`resihp/parallel/reshard.py` 仅被该 gated 测试导入，非 torch 路径不触碰，既有 75 项无回归。

待目标机（带 torch）执行的门禁：`python3 -m pytest -q tests/test_parallel_reshard.py`，Gloo/NCCL 双后端跑同一逻辑（与 T10 结构一致：`_run_reshard`/`_run_boundary` 设备无关，`_gloo_*`/`_nccl_*` 仅切后端与设备）：
- 纯函数（单进程）：peer 拼接 / 复制取一份 / shard 缺失回落 checkpoint / 两处皆无抛 `ReshardError`；`local_slice` 与 `chunk` 一致；`shard_dims` 与 TP 模块分片维逐项相等。
- **CPU/Gloo**（始终可跑）：**`TP2→TP1`**（2 进程，两 shard 拼回、存活 rank 得完整 degree-1 张量、被踢 rank 返回空）；**成员替换 `{0,1}→{0,2}`**（3 进程，rank0 保 shard0、rank2 收 shard1、rank1 退出，**无丢失**，均与 checkpoint `torch.equal`）；**异构边界反向**（上游 TP1→下游 TP2，rank0 边界梯度逐元素等于单进程参考，下游 loss 等于参考）。
- **GPU/NCCL**（`torch.cuda.set_device(rank)` + `cuda:rank` + `nccl`，显式 `assert get_backend()=="nccl"`；GPU 数不足则 skip）：与 Gloo 同三场景在**真实 GPU shard + NCCL 集合**上重跑——reshard 用例断言 shard 确实曾在 GPU（`local_was_device`）后由 NCCL `all_gather_object` 汇总、`_to_cpu` 规范化、正确重构；边界用例断言激活/梯度 `is_cuda` 且梯度逐元素等于参考。`TP2→TP1` 与异构边界需 2 GPU（**2-GPU 服务器即可实跑**，边界特意设计为上游 TP1→下游 TP2 两 rank，`dist.broadcast` 走 NCCL），成员替换需 3 GPU。
通过后本节状态改为「实现完成，待审核」。

遗留问题：
- T11 只交付重切**原语 + 异构边界原语 + 单测**。把它接进安全点⑦ `_recover_state` 的真正收集/重切、分片 TP run 的完整逻辑 checkpoint（optimizer 矩状态 gather），以及六条一致停止条件，属 T12（PP 运行时/层迁移）/T14（原子重配与一致停止）范畴，本任务不动 `control.py`。
- `reshard_tp_state` 假定「每个健康 rank 对每个逻辑张量都贡献其 shard」的纯 TP 汇总；PP 下 stage 只持有非连续层子集时的按-stage 收集在 T12 的层迁移里落地。
- 数值校验用 `torch.equal`（重切是纯搬运/拼接/切片，无浮点归约，逐比特成立）；异构边界前向广播、反向广播亦不引入 reassociation，故梯度可逐元素 `torch.equal` 对齐参考，与 T10 执行路径的 `allclose` 口径不冲突（各自对应无归约/有归约场景）。

## T12：PP 1F1B 运行时与层状态迁移

状态：实现完成，**分布式门禁（Gloo + 真实 GPU/NCCL）待 torch 环境执行**（本机无 torch，计划硬性禁止安装/升级 torch，故 `mp.spawn` 门禁需在带 torch 的目标机跑）。

新增：`resihp/parallel/pp.py`、`tests/test_parallel_pp.py`。未改动 `tp.py`/`reshard.py`/`planner/pp.py`/`model.py`/`control.py`（保持 surgical）。

### 设计要点

- **真正切分模型的 stage**：`PipelineStage` 只持有自己那段连续 layer，首 stage 另持 token/position embedding、末 stage 另持 final norm 与 LM head（计划 3.4「embedding/LM head 归属首/尾可执行 stage」）。子模块命名与参考模型逐字相同，故 `named_parameters()` 直接产出稳定逻辑名（`layers.<gid>.attn.q_proj.weight`），按名从参考 `logical_state_dict` 装载——层迁到别的 stage 后名字不变。测试断言「任一 stage 都不持有整模型」「各 stage 参数集两两不交且并集 == 参考全集」，证明是真 PP 而非元数据。
- **1F1B 是真调度**：`PipelineRuntime.train_step` 按 warmup → 1F1B 稳态 → cooldown 发出计划要求的 Forward/Backward/Send/Recv/WeightUpdate 原语，激活前向、梯度反向都是真实 P2P 传输，micro-batch 梯度累积后一次 AdamW。`schedule` 属性记录发出的原语顺序，测试逐字比对（2 stage/4 micro 下 stage0=`F0 F1 B0 F2 B1 F3 B2 B3 W`、stage1=`F0 B0 F1 B1 F2 B2 F3 B3 W`），锁死「不是先全 F 再全 B」。
- **稳态两处收发必须融合（NCCL 正确性，非性能优化）**：`send_forward+recv_backward` 与 `send_backward+recv_forward` 各用**一次** `dist.batch_isend_irecv` 发出。NCCL 下每个 rank 的传输在自己的 stream 上按序执行，若把这两步拆成先后两次独立收发，stage0 会卡在「送 act1」（对端尚未 post 对应 recv），而 stage1 卡在「送 grad0」（对端被前一步堵住永远到不了 recv）——**确定性死锁**。融合后两个方向同时推进。此点已用离线仿真反证（见下）。
- **loss 按 micro 数缩放**：每个 micro 的 cross-entropy 除以 micro 数后累加，等价于参考的整 batch 均值（各 micro 等大），故梯度与 loss 与参考一致；但 micro 切分重排了 FP32 归约顺序，验收沿用 T10 的 `allclose(rtol=1e-4, atol=1e-5)` 口径，单 micro 时精确。
- **层迁移复用唯一恢复路径，不造第二套机制**：`plan_migration` 把 T4 重分层（哪层落到哪个 stage）与两侧 TP degree 组合成 `LayerPlacement{layer, old_owner, new_owner, old_degree, new_degree}`，张量搬运仍走 T11 的 `reshard_tp_state`（健康 peer 收集 → 与 checkpoint 校验 → 按新布局重切）。`reshard_layout` 给出某 stage 需要重新获取状态的层的分片布局（`shard_dims` 里属首/尾 stage 的 embedding/LM head 不混入层布局）。

### 关键完备性修正（本轮自查发现）

`PPPlan.migrations` 只列**换了 stage** 的层，据此驱动重切会漏掉一整类：**stage 没丢层、但自己掉了一个 TP rank**，其保留的层同样必须从旧 degree 重切到新 degree。实测 `(2,2)/TP(2,2) → TP(1,2)`：`plan.migrations == ()`（无一层移动），但 layer 0/1 的 degree 2→1 必须重切——只看 migrations 会把 stage0 的状态静默留在旧的两路布局里。故 `plan_migration` 对**每个**全局 layer 都给出新旧 `(owner, degree)`，并用 `moved` / `resharded` 两个独立属性区分「换 stage」与「换 degree」（可同时成立）。`test_stationary_layers_still_reshard_when_their_stage_loses_a_rank` 专门锁定这条。

### 本机可做的验证（无 torch）

- `python -m py_compile`（全包 + 全部测试）：通过。
- 全量 `python -m pytest -q`：**75 passed, 6 skipped**——`tests/test_parallel_pp.py` 顶部 `pytest.importorskip("torch")` 在无 torch 时整体跳过（较 T11 基线 5 skipped 增 1）；既有 75 项无回归。
- torch stub 下 `import resihp.parallel.pp`：通过（import 期 torch-free 安全），并在 stub 下跑通纯 planner 路径（`plan_migration` / `reshard_layout` / `balanced_layers`）。
- **离线调度仿真**（scratchpad，纯 Python 复刻调度控制流）：stages×micro = 1×1/1×4/2×1/2×2/2×4/3×3/3×4/4×4/4×8/4×2 全部无死锁，每个 micro-batch 的 F 与 B 各恰一次且按序；4 stage×8 micro 输出标准 1F1B 阶梯。
- **离线运行时仿真**（scratchpad，假 torch/dist + 多线程跑 `PipelineRuntime.train_step` **真身**）：假 `batch_isend_irecv` 复刻 NCCL 群语义（send 必须等到对端 post 对应 recv 才完成，整组全完成才返回）。1/2/3/4 stage 全部跑通、schedule 逐字符合预期、stage forward 恰调用 micro 次、WeightUpdate 恰一次。**反证控制组**：把融合组拆成两次独立单op组后，2 stage 用例如期双向 `TimeoutError` 死锁——证明仿真不是空转、融合设计确实承重。

### 待目标机（带 torch）执行的门禁

`python3 -m pytest -q tests/test_parallel_pp.py`，Gloo/NCCL 双后端跑同一逻辑（`_compare_pp`/`_run_migration` 设备无关，`_gloo_*`/`_nccl_*` 仅切后端与设备）：

- 纯函数（单进程）：`balanced_layers` 连续/唯一/完整与余数前置、stage 多于层被拒；placements 覆盖每层且给出两侧 `(owner, degree)`；「移动且重切」「只重切不移动」「只移动不重切」三类分别锁定；`reshard_layout` 同时含到达层与「原地但降 degree」的层、分片维与 T11 布局一致、不含 embedding/LM head，且对无状态变更的 stage 返回空。
- **CPU/Gloo**（始终可跑）：`test_pp_matches_reference_gloo[1,2]`——2 stage×4 micro，每 stage 的**梯度**与**一步 AdamW 后参数**与参考对应部分 `allclose`，末 stage loss 与参考一致、非末 stage 返回 `None`，参数归属不重不漏，schedule 逐字为 1F1B；`test_pp_layer_migration_is_lossless_gloo`——stage 1 的整份状态工作量（moved+resharded 的 layer 4 与「原地但降 degree」的 layer 3）在一次重切里走真实集合通信，两层共 20 个张量的 `param/grad/exp_avg/exp_avg_sq/step` 全部与 checkpoint 锚点 `torch.equal`，被踢 rank 返回空。
- **GPU/NCCL**（`torch.cuda.set_device(rank)` + `cuda:rank` + `nccl`，显式 `assert get_backend()=="nccl"`、`current_device()==rank`、参数 `is_cuda`；GPU 不足才 skip）：与 Gloo 同两组场景在**真实 GPU 张量 + 真实 NCCL P2P/集合**上重跑；迁移用例另断言 shard 确实曾在 GPU（`local_was_device`）。2 stage 与迁移用例均只需 **2 GPU**，2-GPU 服务器可实跑。

通过后本节状态改为「实现完成，待审核」。

### 复审修正（本轮）

自查发现**同一个完备性缺陷在下游又犯了一遍**：`plan_migration` 已按「每层都给两侧 (owner, degree)」修好，但喂给 `reshard_tp_state` 的布局函数只挑 `moved` 的层，于是「stage 没丢层、只掉了一个 TP rank」的场景又被漏掉。实测 `(2,2)/TP(2,2)→TP(1,2)`：`migration_layout(new_owner=0)` 返回 `{}`，而 layer 0/1 明明需要 2→1 重切——等于把上一层刚修好的坑在下一层重新挖开。修正：函数改名 `reshard_layout` 并按 `moved or resharded` 选层，语义从「到达的层」改为「该 stage 需要重新获取状态的层」；新增 `test_reshard_layout_selects_stage_that_only_lost_a_rank` 专门锁定，分布式迁移用例也随之覆盖「一次重切同时处理 moved 层与原地降 degree 层」。

小修：`train_step` 中四处重复的 `torch.empty(shape, device=device)` 收敛为一个 `buffer()` 局部函数；`exchange` 补注释说明 `ops` 的存活期正是发送缓冲区不被提前回收的原因。

### 遗留问题

- **一次 `reshard_layout` + `reshard_tp_state` 调用只覆盖一个 `old_degree → new_degree` 对**（`reshard_tp_state` 只收单个 `old_size`）。当某 stage 同时接收来自不同 degree 源 stage 的层时，调用方需按 `LayerPlacement.old_degree` 分组多次调用；锁定配置 TP=2 起步、单 rank 逐次失效下不触发，已在 `reshard_layout` docstring 写明。
- **embedding / LM head 的 owner 变更未做状态迁移**：`PPPlan` 给出 `embedding_owner` / `lm_head_owner`，但首/尾 stage 整组失效导致 owner 改变时，这两个张量的搬运不在 `reshard_layout`（其语义是逐层布局，不含非层张量）中，属 T14 接入范畴。
- PP 与 TP 尚未在同一 run 内组合执行：`PipelineStage` 走完整（未分片）层，TP 分片执行在 `parallel/tp.py`。`TP2×PP2` 同时真实执行属 T15/T16 两两组合与端到端范畴。
- 层迁移交付的是**计划 + 复用 T11 重切原语 + 单测**；接进安全点⑦ `_recover_state` 的真正收集/重切链路仍属 T14。`control.py` 本任务未动。
- 1F1B 为**功能调度**（计划 3.4 原话），未做 P2P 与计算重叠、未做 F/B/W 三类的细粒度拆分调度。
- 每个 stage 都收到完整 batch，首 stage 取输入、末 stage 取标签；真实场景的 batch 分发（只发首/末 stage）属 DP 数据路由接入范畴。
