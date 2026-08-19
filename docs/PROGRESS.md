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

待目标机（带 torch）执行的门禁：`python3 -m pytest -q tests/test_parallel_tp.py`（TP1/TP2 各以 `mp.spawn` 起 1/2 进程 Gloo），覆盖：gather 后完整 logits 与 loss 与参考 `allclose`；各 rank 本地 shard 的**梯度**与一步 AdamW 后的**参数**与参考对应切片 `allclose`（degree=1 精确一致，degree=2 容差内）；不整除 degree 被拒；`ControlPlane` 附真实状态后 ② 落盘且可重载、⑧ 摘要为真实逻辑摘要（未附状态时为空串）。通过后本节状态改为「实现完成，待审核」。

遗留问题：
- 数值验收为 `allclose`（容差内），非逐比特；两两组合/端到端（T15/T16）沿用同一口径与参考对照。
- 分布式后端：门禁用 Gloo（CPU）；GPU/NCCL 验收属 T18。
- TP 仅静态执行，无 degree/成员重切；gather+reshard 属 T11。
- `ControlPlane._commit_checkpoint` 目前只能存**完整逻辑** run；分片 TP run 的 checkpoint 需 T11 的 optimizer 矩状态 gather。⑦ `_recover_state` 仍空。
