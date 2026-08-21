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

## T13：DP 跨 replica 执行

状态：实现完成，**分布式门禁（Gloo + 真实 GPU/NCCL）待 torch 环境执行**（本机无 torch，计划硬性禁止安装/升级 torch，故 `mp.spawn` 门禁需在带 torch 的目标机跑）。

新增：`resihp/parallel/dp.py`、`tests/test_parallel_dp.py`。未改动 `planner/dp.py`（T5）/`pp.py`（T12）/`tp.py`（T10）/`reshard.py`（T11）/`control.py`/`model.py`/`reference.py`（保持 surgical）。

### 关键设计决策（计划 3.5 逐-stage 语义 vs T5 整-replica assignment 的歧义）

计划 3.5 运行时保证写「失效 stage 的 micro-batch → 健康 peer **stage**」（逐 stage 跨 replica），而已锁定的 T5 `planner/dp.py` 实际按**整 replica** 迁移（一个 micro 始终在单一 replica 内）。T13 指令是「把 T5 的 assignment **接到真实执行**」。

- **决策**：运行时**完全由 assignment 驱动**——每个 `(micro, stage)` 的 executor、上/下游 peer 都从 `DPAssignment.placements` 查表，**绝不从固定拓扑推导**（计划 3.2 硬性要求）。因此它对 T5 的整-replica assignment 与手工构造的逐-stage 跨-replica assignment **都正确**，忠于计划 3.5 原文又不与 T5 冲突。`executor_route(assignment, rank)` 是纯查表函数，`crossreplica` 场景下 rank0 的下游随 micro 变化（micro0/1→rank1、micro3→rank3，跨到另一 replica），单测直接锁定。

### 设计要点（计划 3.5 六条运行时保证逐条落地）

- **① 每 micro·stage 恰执行一次**：运行时只跑 assignment 把本 rank 列为 executor 的 `(micro, stage)`；测试对全体 rank 的 `processed` 断言两两不交且并集完整（无 workload 同归属源与目标，保证④）。
- **② forward 发给实际下游 / backward 返回实际上游**：`train_step` 用 T12 同款融合 `batch_isend_irecv` 收发，peer 取自 `route["downstream"]/["upstream"]`（即 assignment 里下/上一 stage 的 executor）——rerouted micro 的激活因此跨 replica 传到真正的下游 executor，梯度回到真正的上游 executor。全前向→全反向的最简 DP 调度（1F1B 是 PP/T12 职责，不重做），已手工反证无死锁（前向 DAG 严格 stage0→stage1，反向反序配对）。
- **③ activation 生命周期**：`ActivationLog` 在 forward `retain`、对应 backward `release`；`live` 永不含已完成 backward 的激活，`peak` 为并发驻留高水位。纯单测锁定「retain/release/峰值/对已释放者再 release 报错」，运行时门禁另断言 `peak == len(processed)` 且结束 `live` 清空。
- **⑤ executor 改变不影响 global batch 归一化**：每个 micro 的 loss 除以**全局** micro 数后累加，各 replica 持自身 micro 的梯度**偏和**；DP 合并对每个逻辑参数**跨 replica 求和**恰得整-batch 均值——3-vs-1 失衡切分（模拟 reroute）下 AdamW 更新仍逐参与参考一致。
- **⑥ 不同 replica 的 PP 分层 / TP degree 可不同**：`dp_combine_gradients` 统一走「**每 replica 用 T11 `reconstruct_full` 重建完整逻辑梯度 → 跨 replica 求和 → 用 `local_slice` 按各自 degree 重切**」**单一路径**（不造第二套 all-reduce）。同一函数同时覆盖同构、PP 异构（replica A 两 stage / replica B 单 stage 持全层）、TP-degree 异构（replica A TP2 / replica B TP1）；`shard_dim=None` 的复制参数只取一份，避免 TP 组内复制副本被重复计数（离线单测专门反证）。old_size 按 **replica** 记录，修掉「用全局单一 old_size 重建异 degree replica」的隐患。

### 本机可做的验证（无 torch）

- `python -m py_compile resihp/parallel/dp.py tests/test_parallel_dp.py`：通过。
- 全量 `python -m pytest -q`：**75 passed, 7 skipped**——`tests/test_parallel_dp.py` 顶部 `pytest.importorskip("torch")` 在无 torch 时整体跳过（较 T12 基线 6 skipped 增 1）；`resihp/parallel/dp.py` 仅被该 gated 测试导入，既有 75 项无回归。
- **离线逻辑仿真**（scratchpad 假 torch/dist + 1-D FakeTensor 跑函数真身）：`executor_route` 跨-replica 路由（rank0 下游随 micro 变化、rank3 跨界收 micro3）；`ActivationLog` 生命周期与非法 release；`dp_combine_gradients` **TP-degree 异构**（TP2 两 shard + TP1 全量，和为参考、各 rank 得正确重切）、**PP 异构/复制形态**求和、**复制参数不重复计数**（TP 组内两份相同副本只入一份，5+5+1 ≠ 6 的反证）全部通过。

### 待目标机（带 torch）执行的门禁

`python3 -m pytest -q tests/test_parallel_dp.py`，Gloo/NCCL 双后端跑同一 runner（`_run_*` 设备无关，`_gloo_*`/`_nccl_*` 仅切后端与设备），验收口径沿用 T10/T12 的 `allclose(rtol=1e-4, atol=1e-5)`（micro 切分重排 FP32 归约）：

- 纯函数（单进程）：`executor_route` 读实际上/下游 executor（含跨-replica）；`stage_pipeline` 按 stage 排序；`ActivationLog` 生命周期。
- **CPU/Gloo**（始终可跑）：
  - `dp_normalization`（2 rank，全模型两 replica，3-vs-1 失衡）——各 rank 梯度/一步 AdamW 后参数与参考 `allclose`、loss 汇总等于参考、activation 峰值=处理数且结束清空。
  - `cross_replica`（4 rank，PP2×DP2，micro3 的 stage0/stage1 落在不同 replica）——断言 rank0 对 micro3 的下游确为另一 replica 的 rank3，激活/梯度真实跨界，各 stage 合并梯度与参考一致。
  - `pp_heterogeneous`（3 rank，replica A 两 PP stage / replica B 单 stage 持全层）——异构 PP 分层合并后与参考一致。
  - `tp_heterogeneous`（3 rank，replica A 真实 TP2 分片前反向 / replica B TP1）——DP 合并重建+重切后各 rank shard 与参考对应切片一致。
- **GPU/NCCL**（`torch.cuda.set_device(rank)` + `cuda:rank` + `nccl`，显式 `assert get_backend()=="nccl"`、`current_device()==rank`、`is_cuda`；GPU 不足才 skip）：与 Gloo 同四场景在真实 GPU 张量 + 真实 NCCL P2P/集合上重跑。`dp_normalization` 仅需 **2 GPU**（2-GPU 服务器可实跑）；`pp_heterogeneous`/`tp_heterogeneous` 需 3 GPU、`cross_replica` 需 4 GPU，不足则 skip。

通过后本节状态改为「实现完成，待审核」。

### 复审修正（本轮）

- **简洁性**：删掉 `ActivationLog.history`（记录每次 retain/release 事件的列表）——无任何代码或测试读取，属投机状态。生命周期由纯单测（retain/release/峰值/非法 release）+ 运行时门禁（`peak == 处理数` 且结束 `live` 清空）验证，`history` 冗余。删后离线仿真与全量 75 passed 无回归。
- 复审确认无正确性/完备性问题：跨-replica micro（stage0/stage1 落不同 replica）的前向/反向梯度经手工逐 rank trace 与参考一致；`dp_combine_gradients` 的「每 replica 按自身 degree 重建→跨 replica 求和→重切」是失衡/异构下的**根本正确**归一化（非兜底），且 `total = full.clone()` 已隔离 `reconstruct_full` 对 `shard_dim=None` 分支返回未克隆张量的别名风险；全前向→全反向的最简 DP 调度经 P2P 逐 micro 配对分析无死锁。

### 目标机（torch + GPU）门禁执行修正（本轮）

- 目标机实跑 `tests/test_parallel_dp.py`（Gloo 全部 + NCCL 按 GPU 数）：**数值全部通过**——`dp_normalization`/`cross_replica`/`pp_heterogeneous` 三场景各 rank 的梯度/一步 AdamW 后参数与参考 `allclose`，实测 `max_grad_diff ~1e-8…1e-9`、`max_param_diff ~1.5e-8`；`tp_heterogeneous`（含真实 TP2 分片）合并后与参考切片一致，Gloo/NCCL 均 PASS。运行时逻辑经真实多进程 + 真实 NCCL 验证正确。
- **修一个纯测试 harness bug**：`_assert_runtime` 的 workload 互斥检查 `set(pairs)` 报 `TypeError: unhashable type: 'list'`——`processed` 里的 `(micro, stage)` 元组经 result JSON 文件往返被反序列化成 **list**（不可哈希）。修复：断言前 `tuple(pair)` 归一。仅测试断言改动，运行时代码与数值不受影响。

### 遗留问题

- 运行时按**全前向→全反向**最简 DP 调度，无 1F1B 交叠（PP 交叠是 T12 职责，本任务不重做）；`DataParallelRuntime` 只驱动 **TP 未分片**（TP1）的 `PipelineStage` replica，TP 分片前反向仍是 T10 `TensorParallelTransformer` 的职责。`tp_heterogeneous` 门禁只对**DP 合并**用真实 TP2 分片梯度验收；「同一 run 内 TP×DP 端到端」属 T15 两两组合。
- 运行时消费 T5 assignment 但门禁多用**手工构造**的 assignment 以覆盖逐-stage 跨-replica（T5 现产整-replica）；把 T5 逐-stage reroute 与运行时对接、接进安全点⑦ `_recover_state` 的真正收集/重切链路属 T14。`control.py` 本任务未动。
- 数值验收 `allclose`，与 T10/T12 同口径；完整 3D `TP2×PP2×DP2` 8 进程端到端属 T16。

## T14：原子重配与一致停止

状态：实现完成，**分布式门禁（Gloo + 真实 GPU/NCCL）待 torch 环境执行**（本机无 torch，计划硬性禁止安装/升级 torch，故 `mp.spawn` 门禁需在带 torch 的目标机跑）。

新增：`resihp/recovery.py`、`tests/test_recovery.py`。改动：`resihp/parallel/tp.py`（TP 模块参数化为**唯一的 stage 类**）、`resihp/parallel/pp.py`（删掉重复的 `PipelineStage`）、`resihp/parallel/dp.py`（运行时接真实 TP stage + TP 感知的 stage 边界）、`resihp/parallel/reshard.py`（donor 按分片 degree 分组 + `shard_logical_state`）、`resihp/control.py`（安全点②/⑦真正落地 + 一致停止协议 + 按 plan.stages 建组）、`resihp/checkpoint.py`（+内容摘要、+`load_anchor`）、`resihp/train.py`（入口跑真实 3D + 停止即正常退出）、`tests/test_control.py`、`tests/test_parallel_{tp,pp,dp,reshard}.py`（随 stage 类统一而更新）。

### T14 是编排层，不是第二套 PP/DP

唯一恢复路径**完全由新 `ExecutionPlan` 驱动**，每个环节交给它的属主模块：

| 关注点 | 来源 | 由谁执行 |
|---|---|---|
| TP 成员 / degree | `StagePlan.tp_members` / `tp_degree` | T11 `reshard_tp_state`（健康 peer 优先 → checkpoint 兜底 → 按新 degree 重切） |
| stage 层归属与变化 | `StagePlan.layer_range`（新旧两版计划） | T12 `LayerPlacement` / `reshard_layout` |
| micro-batch·stage executor | `ExecutionPlan.placements` | T13 `DataParallelRuntime`（`dp_assignment(plan)` 直接把计划的 placements 还原成 `DPAssignment`） |
| 训练通信组 | `plan.stages` / `plan.active_ranks` | `ControlPlane.build_training_groups` |
| stage 实例化 | 上面三者的组合 | T10 `TensorParallelStage` |

`recovery.py` 里没有任何前反向、调度或通信原语；`ControlPlane.training_step()` 就是 `PlannedRun.step()` → `DataParallelRuntime.train_step()`。**不存在 PP=1 假设**：布局一律由 `stage_layout(plan, stage)` 从 `layer_range` 推出，embedding / LM head 归属该 replica 的首 / 末**可执行** stage。

### 唯一的 stage 类（本轮最大的结构改动）

改前 T10 的 `TensorParallelTransformer` 是「TP 分片但整模型」，T12 的 `PipelineStage` 是「层子集但不分片」——两者的 forward 逻辑重复，且**没有任何类能同时表达「PP 层子集 + TP 分片」**，这正是上一版 T14 只能退化成「每 rank 整模型」的根因。

改法：把 T10 的类参数化为 `TensorParallelStage(config, *, vocab_size, sequence_length, layer_ids, is_first, is_last, local_state, group)`，删除 `PipelineStage`，`PipelineRuntime` / `DataParallelRuntime` 一律驱动它。整模型 TP run = 「拥有全部层且首尾皆是」的 stage；TP degree 1 的流水 stage = 同一个类跑在单 rank 组上（T10 已证明 degree 1 时集合通信是 no-op、与参考逐比特一致）。**不保留兼容层**：`source_state`（构造函数内部分片）改为 `local_state`（**本 rank 的分片**，正是 `reshard_tp_state` 的返回形状），调用方需要时用新增的 `shard_logical_state(full, layout=, tp_rank=, tp_size=)` 自己切——于是「stage 永远按当前计划给的布局实例化」是构造函数层面的事实，而不是约定。

### 恢复链路（安全点②/⑦/⑧）

- **②提交 checkpoint**：全体 rank 用整模型布局做一次 `reshard_tp_state(new_size=1)`，各 stage / 各 replica 的分片并起来就是完整逻辑态；仅 rank 0 原子落盘，磁盘恒为唯一文件。
- **⑦恢复**：`load_anchor` 读 checkpoint → `acquire_layout` 算出**本 stage 必须重新获取**的名字 → `reshard_tp_state` 按新 degree 取回 → 与「本 rank 已持有且布局未变」的名字合并 → 用 `TensorParallelStage` 按 `layer_range` + TP 组实例化 → 用 `DataParallelRuntime` 按 `plan.placements` 建运行时 → 装回 AdamW 矩状态与 `cursor`。
- **`acquire_layout` 就是 T12 的迁移视图**：`LayerPlacement` 由**新旧两版计划**读出（不重跑 `repartition_pp`，避免出现第二个布局权威），交给 `reshard_layout` 选出「换了 stage **或** 换了 degree」的层；`reshard_layout` 按设计不含 embedding / LM head，而首 / 末可执行 stage 变更时这两者的 owner 会变——**这正是 T12 明确留给 T14 的口子**，在此补齐。没被选中的名字本 rank 已按正确布局持有，**一个字节都不搬**（`active_ranks` 单调收缩这一计划不变量保证了「degree 不变 ⇒ 成员与分片下标都不变」，已写进 docstring）。
- **⑧校验**：把**实际装好**的分片再 gather 回完整逻辑态，与 anchor 逐张量 `torch.equal`（计划 3.3 第 5 步 = 停止条件「状态重切后完整逻辑张量不一致」），其摘要即步骤⑧的 state digest，因构造相同故各 rank 天然一致。⑦的 acquire 那次重切因此不再重复校验（`verify=False`），不做两遍。

### 六条一致停止条件

`STOP_CODES` 六项：`no_feasible_tp` / `no_executable_pp` / `no_feasible_dp_target`（planner 原码原文透传）+ `checkpoint_unusable` / `plan_disagreement` / `state_mismatch`（控制面自己观测）。

- **共识而非本地判定**：`ControlPlane.agree(reason, **digests)` 在**常驻控制组**上 `all_gather_object`，任一 rank 报出的原因、或任一摘要不一致，都让**每个** rank 抛出**同一个** `ConsistentStop`——原因取自 gather 到的列表（最低报告 rank / 摘要集合），不取本地视角，否则「rank 3 与其余不一致」会让双方各报各的。
- **计划摘要必须在建组之前对**：计划 3.2 把摘要校验放在第⑧步，但**用互不相同的计划去建组本身就是死锁**（`new_group` 是全局集合调用且成员表不同）。所以第⑤步重规划后立刻 `agree(reason, plan=digest)`；第⑧步再对 plan + state 双摘要。同一个 `agree`，不是第二条路径。
- **无半完成组/计划**：⑤处停止时新组根本没建；⑦/⑧处停止时 `except ConsistentStop` 先 `destroy_training_groups()` 再抛。门禁统一断言「组要么原封不动是故障前那对，要么已销毁」。
- **故障前 checkpoint 完好**：停止路径在第②步之后不再写盘；除 `checkpoint_unusable`（注入手段本身就是删掉它）外均断言仍可 `load_checkpoint` 回 `(1, 0)`。
- **只指根因**：`_recovery_stop` 只把 `CheckpointError` / `ReshardError` 归为停止条件，其它异常原样上抛——那是 bug 不是资源条件。

### 其它必要改动

- **训练组按计划建两类**：每个 active stage 一个 **TP 子组**（stage 分片执行用），外加一个覆盖 `plan.active_ranks` 的 **executor 组**（assignment 驱动的流水 P2P 与 DP 梯度合并用）。全体 rank 按 `(replica_id, stage_id)` 顺序对每组调 `new_group`，只留自己那个。
- **`DataParallelRuntime` 的 stage 边界改为 TP 感知**：原实现 `route["downstream"][0]` 只发给下游 stage 的第一个 executor——TP degree > 1 时其余 TP rank 收不到激活。改为「两侧 **leader** 之间搬**一份**权威副本 + 接收方在自己的 TP 组内 `broadcast`」：既不像 all-reduce 那样重复累加，也不会让非 leader 挨饿；反向同理（stage 输入的梯度经 `_CopyToRegion` 已在 TP 组内 all-reduce 过，各 TP rank 一致，leader 直接回送即可）。degree 1 时 broadcast 跳过，行为与改前逐字相同。`executor_route` 相应多返回 `executors`（本 `(micro, stage)` 的整组 executor）。
- **`_combine_and_step` 用真实分片元数据**：原来硬编码 `shard_dim=None, old_size=1`（只对 TP1 成立）；改为取自 stage 的 `local_shards()` / `tp_rank` / `tp_size`——`dp_combine_gradients` 本就按「每 replica 重建完整逻辑梯度 → 跨 replica 求和 → 按各自布局重切」写成通用形态，这里只是把真实值喂进去。`self.rank` 也由 `get_rank(group)` 改为**全局** rank：assignment 说的就是全局 rank，传子组时原写法会查错。
- **异 degree donor 分组**（承上一版）：贡献按 `(shard_count, shard_index)` 归档，`_reconstruct` 按确定顺序挑第一个完整 donor 集。单 rank 逐次失效下 replica A 已是 TP1、replica B 仍是 TP2 时，degree-1 的 index 0 是整份、degree-2 的 index 0 是一半，平铺合并会拼出乱码。
- **`reconfigure` 接入显存门**；**checkpoint 内容摘要 + `load_anchor`**（缺失/损坏/摘要不匹配分别指名根因，摘要校验放在结构校验**之后**，让「缺参数/形状错/游标不一致」仍报各自根因）。
- **入口 `resihp/train.py` 跑真实 3D**：建初始计划 → 建组 → `initial_run` 按计划实例化本 rank 的 stage → 循环 `training_step()`；`VOCAB_SIZE` / `SEQUENCE_LENGTH` 作为运行常量（训练配置 schema 由 T1 锁定，不含这两项）。`ConsistentStop` 打印结构化根因后正常退出。
- **删除被本次改动孤立的符号**：`PipelineStage`、`ControlError`、`agree_on_digest`（并入 `agree`）、`_state_digest()`（改为 `state_digest` 属性）、`training_step(plan)` 的 T9 all-reduce 占位分支。

### 本机可做的验证（无 torch）

- `python -m py_compile`（全包 + 全部测试）：通过。全量 `python -m pytest -q`：**75 passed, 8 skipped**，既有 75 项无回归。
- **离线仿真 1**（假 torch，1-D FakeTensor 跑 `_merge`/`_reconstruct` 真身）：同构 degree-2 拼接；**异构 degree（1 与 2 并存、degree-2 缺 index 1）落到 degree-1 完整集**；全缺失落 checkpoint；两处都无则抛 `ReshardError`；旧格式按 `old_size` 归档；复制张量任取一份。全过。
- **离线仿真 2**（线程扮演 rank + 假 `torch.distributed`，跑 `ControlPlane.safe_point` 真身，集合通信按真实 rendezvous 语义实现）：四类停止（`no_feasible_tp` 走真实 `build_plan` 显存门、`plan_disagreement` 只改 rank 1 的重规划、`checkpoint_unusable`/`state_mismatch` 在⑦处抛）**全部：无 rank 阻塞、码与消息各 rank 完全一致、`published == []`、组不残留**；逐类核对建组次数——⑤处停止时新组**一次都没建**，⑦处停止时新组建了又被销毁。健康路径：一个新版本、摘要唯一、只有幸存 rank 拿到新组。
- **离线仿真 3（本轮新增）**：线程扮演 4 个 rank，假 P2P 按**严格 rendezvous**（送出必须等到对端 post 对应 recv 才完成，即 NCCL 侧最保守的模型）跑 `DataParallelRuntime.train_step` **真身**，拓扑 `TP2 × PP2`。结果：**无死锁**；各 rank 恰好执行 assignment 给它的 `(micro, stage)`；首 stage 走 `tokens=` 分支、后续 stage 走 `hidden=`；activation 峰值 2、结束清空；每 rank 恰一次 WeightUpdate；**P2P 只由两侧 leader（rank 0 与 rank 2）承担**，TP peer 靠组内 broadcast 拿到副本——即每个边界跳只有一份权威副本。
- **planner 层预期值核对**（真实 `build_plan`，无 torch）：`PIPELINE`（TP2×PP2×DP1，6 层）掉 rank 1 → stage0 degree 2→1 且层 0-2 缩为 0-1，**层 2 迁到 stage1**（2-5）；`REPLICATED`（TP2×PP1×DP2）掉 rank 1 → replica0 degree 1 而 replica1 不变（**未受影响的 replica 一个字节不搬**），再掉 rank 3 → 两 replica 在事件②时 degree 各为 1 与 2（异构 donor 路径）；`test_control` 的 8 进程 3D 布局逐 rank 层归属亦逐项核对。
- 另外确认：`tp=4` 且连续两次故障会命中 T6 的 `active ranks must shrink monotonically` 不变量（`choose_tp` 会想启用一个先前 idle 的 rank），`build_plan` 直接抛 `PlanInvariantError`。它在各 rank 上确定且一致，不会脑裂；但它既不是计划也不是六条停止之一，记为遗留。

### 待目标机（带 torch）执行的门禁

`python3 -m pytest -q tests/`（重点 `tests/test_recovery.py`、`tests/test_control.py`，以及随 stage 类统一而改动的 T10/T11/T12/T13 用例），Gloo/NCCL 双后端跑同一 runner（控制组两种后端下都是 Gloo——它必须常驻且要跑对象集合；只有训练组切 NCCL）：

- **纯函数**：`stage_layout` 按 `layer_range` 给出层归属、embedding/LM head 落在首/末可执行 stage、两 stage 不相交；`acquire_layout` 在 `PIPELINE` 事件下对 stage0 给出「层 0,1 + embedding」、对 stage1 给出「只有迁入的层 2」且真子集于其拥有集；未受影响的 replica 取到 `{}`；peer 与 checkpoint 同时可用时 **peer 胜出**；checkpoint 缺失 / 不可读 / 篡改后摘要不匹配分别指名根因；三个 planner 码由真实 planner 抛出。
- **恢复（4 rank）**：`pipeline`（TP2×PP2×DP1）——一次事件同时覆盖 TP 重切、**真实 PP 层迁移**、embedding 重切、以及「原地不动因而不搬」的层；`replicated`（TP2×PP1×DP2）——连续两次 fail-stop，事件①从 peer replica 收集且该 replica 自身零搬运，事件②两 replica degree 不同。两者都断言：恢复后各 rank **持有的名字恰好是其新 stage 该拥有的**（不多不少，即不再是每 rank 整模型），且逐张量 `torch.equal` 等于 checkpoint 在新 degree 下的切片（原则 A 恢复前口径）；被踢 rank 不持有任何状态；每事件只产生一个新版本；恢复后继续训练；最后 checkpoint 可重载。
- **停止（2 rank，六条各一个用例，参数化）**：全体同码同消息 / 无新计划外泄 / 无属于被否决计划的活组 / 无残留进程组 / 故障前 checkpoint 可重载 / 消息不提第二个条件；并用带超时的 `mp.spawn(join=False)` 轮询确保**超时前全体退出**而不是卡在集合通信里。
- **GPU/NCCL**：与 Gloo 同全部场景，在真实 GPU 张量 + 真实 NCCL 训练组上重跑（断言 `torch.cuda.current_device()==rank`、参数 `is_cuda`、`get_backend(tp_group)=="nccl"`）。恢复用例需 **4 GPU**，六条停止用例只需 **2 GPU**。
- **回归**：`tests/test_control.py` 现在是真实 `TP2×PP2×DP2` 8 进程运行（每 rank 只持自己 stage 的层），仍锁定「每事件一个版本 / 失效 rank 永久退出 / 各 rank 摘要一致 / 无残留组」。

通过后本节状态改为「实现完成，待审核」。

### 遗留问题

- ~~`no_executable_pp` / `no_feasible_dp_target` 不可达~~ —— **已解决**，见下节「遗留问题清算」第 1 条。
- ~~`tp≥4` 连续故障撞上 `PlanInvariantError`~~ —— **已解决**，见下节「遗留问题清算」第 2 条。
- **`PipelineRuntime`（T12 的 1F1B）未被控制面使用**：安全点接的是 T13 的 assignment 驱动运行时（micro-batch·stage executor 是它的职责）。1F1B 与 assignment 路由在同一 run 内合流属 T15 两两组合。
- **`_recovery_stop` 只认两类异常**：其它异常照常上抛，此时该 rank 会离开而其余 rank 仍等在 `agree` 上——门禁的超时轮询会判失败而不是无限挂起。这是刻意的：把任意 bug 包装成「干净停止」会掩盖问题。
- 每个安全点有三次全局对象 gather（②提交、⑦获取、⑧校验），各自对应计划里一个明确步骤，未做合并；本复现模型极小，未做性能优化。

### 遗留问题清算（本轮：两条遗留全部解决，不再推给 T15/T16）

#### 1. `no_executable_pp` / `no_feasible_dp_target` 不可达 —— 结论：**是 build_plan 掩盖了根因，非语义冗余**

逐条分析后，两者都属「前面的 planner 仍可行、当前 planner 独立不可行」的**合法输入**，此前不可达是 `build_plan` 的提前过滤与状态传递造成的：

- **`no_executable_pp`**：`if not any(new_tp): continue` 把「整组失效的 replica」静默跳过，并在**所有** replica 都死时另起炉灶报一个自造码 `no_surviving_replica`（该码甚至不在 `STOP_CODES` 里）。修正：把被清空的 replica 的 `(old_layers, old_tp, new_tp)` 记下来，循环结束若无任何存活 replica，就把它交回 `repartition_pp`——全零 capacity 下它必然抛出自己的 `no_executable_pp`。自造码 `no_surviving_replica` 随之删除，`STOP_CODES` 六项恰好等于可达集合。
  - 附带修正（本轮自查发现的**新缺陷**）：第一版把 `wiped_out[0]` 无条件取第一个记录，而**上一轮事件就已清空**的 replica 在 `old_layout` 里没有任何层，`repartition_pp` 会以 `ValueError: old_stage_layers must contain at least one layer` 拒绝**输入**而不是报告条件——又一个未分类逃逸。改为只记录 `any(old_tp)` 的 replica：本次事件才被打空的那个才是证据。实测 `TP2×PP1×DP2` 依次杀 0→1→2→3 现在给出 `no_executable_pp`。
- **`no_feasible_dp_target`**：`build_plan` 建 `DPTopology` 时**根本没传 `memory_budget`**，于是计划 3.5 明写的 `MemoryFeasible` 判定从未运行。这不是冗余检查：`choose_tp` 只能用每个 stage 的**旧**层数把关（重分层需要新 degree，顺序上必须在后），而 `repartition_pp` 可以把**更多**层交给某个 stage——TP 门放行的和最终发布的不是同一个布局。数值搜索确认存在这样的输入（`L=4, pp=2, tp=2, dp=1`，杀 2 再杀 3：TP 只按 3 层校验过，最终 stage 要装 4 层）。修正：把 `memory_budget` / `sequence_length` / `vocab_size` / `in_flight_micro_batches` 透传给 `DPTopology`，并把 `InfeasibleDP` 包成 `InfeasiblePlan`（此前**根本没包**，会以裸异常逃逸）。DP 于是成为唯一看得见最终布局的关口，正是计划 3.5 的原文。
- 六条停止条件与当前 planner 模型**不冗余**：三个 planner 码各自对应一个真实且互不蕴含的失败面（degree 无解 / 无流水可排 / 最终布局装不下）。

#### 2. `active ranks must shrink monotonically` —— 语义错误，已换成真正的 fail-stop 表述

最小复现（`TP4×PP1×DP1`，world 4）：

```
v0  failed=()      active=(0,1,2,3)  members=(0,1,2,3)
v1  failed=(0,)    active=(1,2)      members=(1,2)      <- 健康的 rank 3 被闲置
v2  planner 想要   members=(2,3)                        <- rank 3 从未失效，只是上一版没被选中
    assert_invariants: active(2,3) ⊄ previous.active(1,2)  -> PlanInvariantError
```

根因：`choose_tp` 取存活者里**最大 2 的幂前缀**，因此会合法地闲置健康 rank；旧不变量把「本版未被使用」等同于「永久失效」，于是禁止了完全合法的重新启用。

修正后的语义：`failed_ranks` 严格累计（`previous.failed ⊆ failed`），且 `active ∩ failed == ∅`——**两者合起来就是「失效 rank 永不重回训练路径」**，同时不再限制健康闲置 rank 被重新选中。

- 关于「换一条不变量」而不是删除：先写成 `active ⊆ previous.live_ranks`，随后证明它**恒真、不可达**——`active ∩ failed == ∅` 且 `previous.failed ⊆ failed` 即可推出 `active ∩ previous.failed == ∅`，写不出任何能触发它却通过前两条的计划（回归测试也因此写不出来）。为避免留下一行永不执行的断言，改为**删除该行并把保证写在承载它的两条检查处**；回归测试从两条**可达**路径钉死：把 rank 从 failed 里摘掉 → `failed ranks must grow`；留在 failed 里却仍排它上阵 → `active and failed ranks overlap`。

#### 3. 上述改动连带暴露并修掉的两个真实缺陷

- **`recovery.acquire_layout` 的保留规则不再成立**：它原先依赖被删掉的那条不变量，论证「degree 不变 ⇒ 成员与分片下标不变」。允许重新启用后，`(1,2)@2 → (2,3)@2` 里 rank 2 的 shard index 从 1 变 0、rank 3 全新入座——前者会**静默保留错误的分片**（只能在第⑧步被当成 `state_mismatch` 误报），后者直接 KeyError。改为显式判据 `_keeps_its_shard(previous, plan, stage, rank)`：只有「同 stage、同 degree、同 index」才算已持有正确形状，否则整段重取。边界条件里的 degree 比较随之成为死代码，一并删除。
- **checkpoint writer 写死 rank 0**：`writer=self.rank == 0` 且 cursor 取自 rank 0 的 run。累计故障下 rank 0 本身可能已失效而不再持有状态，于是会写出 `completed_steps=0` 的**损坏 checkpoint**（下次 load 报「游标不一致」）。改为 `writer=self.rank == plan.active_ranks[0]`——各 rank 从同一计划推出、且必然是一个真正持有状态的 rank。
- **cursor 来源**：新入座的 rank 没有自己的 cursor 可继承。`load_anchor` 改为返回 `(anchor, completed_steps)`，恢复一律以 checkpoint 的迭代号为准（它本就是计划 3.1 定义的恢复点与数据游标），特判随之消失。

#### 本轮验证

- 全量 `python -m pytest -q`：**81 passed, 8 skipped**（较上轮 75 增 6 项回归用例）。
- **分类完备性扫描**：6018 条故障顺序（world ≤ 4 穷举、world 8 定种子抽样），有/无显存预算各跑一遍，共 **19884 次合法重规划**；停止分布 `{no_executable_pp: 3018, no_feasible_dp_target: 30, no_feasible_tp: 2970}`，**未分类逃逸 0 次**。
- 三个离线仿真（异 degree donor 分组 / `safe_point` 一致停止 / `TP2×PP2` 的 `DataParallelRuntime` 严格 rendezvous 收发）全部无回归。
- 新增回归用例：`tests/test_plan.py` —— 全灭由 PP planner 命名、早已清空的 replica 不作为证据、DP 拒绝装不下的最终布局、失效 rank 两条复活路径均被拒、健康闲置 rank 可被重新选中、累计故障序列不掉出六条分类、同故障序列摘要逐版本一致；`tests/test_recovery.py` —— 失去座位的 rank 一律整段重取、三个 planner 码由**真实重规划**产生（同时校验 `STOP_SCENARIOS` 表诚实）。
- 六条一致停止的分布式门禁现在**全部走真实 `safe_point → build_plan`**：`no_executable_pp`（world 2，杀 1 再杀 0）与 `no_feasible_dp_target`（`TP1×PP2×DP1`，stage 1 死后 stage 0 吃下全部层并越过预算）不再注入 reason 对象；仅 `plan_disagreement` 仍注入——确定性重规划无法自己和自己不一致，被测的是控制面的反应。
- 新增恢复门禁 `*_reseat`（`TP4×PP1×DP1`，连杀 0、1）：断言健康闲置 rank 3 被重新启用、rank 2 与 rank 3 在事件②整段重取、且所有在座 rank 恢复到**同一个 cursor**。

### 目标机门禁执行修正（3 failed → 根因与修复）

#### A. `checkpoint_unusable`（Gloo + NCCL 同时失败）——场景前提自相矛盾 + 两个 loader 契约不一致

- **场景前提错**：注入手段是**删除唯一的 checkpoint**，同时又断言「停止后最后有效 checkpoint 可重载」。计划 3.6 明写「只保留当前恢复所需的**唯一最新** checkpoint」，所以「该 checkpoint 不可用」与「还有一个有效 checkpoint 可重载」在同一条件下不可能同时成立。改为**篡改**而不是删除：文件仍可 `torch.load`，被拒的是**存储的内容摘要**（正是 T14 加的那一项），且留下一个可以被证明「停止路径没碰过」的文件。断言相应改为该条件真正欠下的保证——`tmp_left` 为假（写入始终原子）、文件字节与故障注入后**逐字节一致**（停止路径没写、没换、没删）、且根因确为「摘要不匹配」。**没有跳过 reload 断言**：其余五条仍断言 `[events, events-1]` 可重载；只有这一条的**主语就是那唯一的 checkpoint**，故换成更强的原子性断言并在测试里写明理由。
- **顺带暴露的实现缺陷**：`load_anchor` 会检查文件是否存在并抛 `CheckpointError("checkpoint 缺失")`，而 `load_checkpoint` 直接 `_torch_load`，漏出裸的 `FileNotFoundError`——同一个模块的两个 loader 对「坏文件」给出两种契约，调用方得同时 catch 两类异常。抽出**唯一**的 `_read_payload(path)`（缺失 / 不可读各自指名根因）供两者共用。`tests/test_checkpoint.py` 10/10 不受影响（它们篡改的是已存在的文件）。

#### B. `test_recovery_cuda_nccl[pipeline]` 挂死——batched P2P 跑在**整组的集合通信器**上

根因（torch `ProcessGroupNCCL::pointToPoint`）：

```
batchP2P = coalescing_state_ & CoalActive          # batch_isend_irecv 会置位
if batchP2P: key = getKeyFromDevice(device)        # -> 该 group 的集合通信器
else:        key = getKeySendRecv(rank, peer)      # -> 只含两个 peer 的通信器
```

`DataParallelRuntime._send/_recv` 用 `batch_isend_irecv`（每次只有**一个** op），于是边界 P2P 落在 **executor group 的集合通信器**上。该通信器上各 rank 的入队顺序变成：

| rank | executor-group 通信器上的入队顺序 |
|---|---|
| 0 | send(m0), send(m1), recv(g m1), recv(g m0), all_gather |
| 1 | all_gather |
| 2 | recv(m0), recv(m1), send(g m1), send(g m0), all_gather |
| 3 | all_gather |

NCCL 要求**同一通信器上所有 rank 按相同顺序入队**。rank 1/3 完全没有那四个 P2P，于是它们的 `all_gather` 与 rank 0/2 的 P2P launch 配对 → 挂死。

**修复**：边界 hop 改用**非批量**的 `dist.isend` / `dist.irecv`（每次本来就只有一个 op，批量毫无意义）。非批量 P2P 拿到的是「只含两个 peer」的通信器，executor group 的集合通信器上只剩 DP 合并的 `all_gather`，四个 rank 顺序一致。

**为什么不影响已通过的路径**：

- `replicated` / `reseat`：`pp=1`，单 stage 既是首也是尾，`_send`/`_recv` **一次都不会被调用**，零影响。
- Gloo 的 `pipeline`：`batch_isend_irecv` 的非 CUDA 分支本身就是逐个调用 `isend`/`irecv`；单 op 时新旧写法走的是**同一段代码**，逐字等价。
- T12 `PipelineRuntime` **未改**：它融合的是 send+recv **两个** op，批量是必需的（其 docstring 已论证拆开会确定性死锁）；其 group 本就恰好是流水线的那几个 rank。已在 docstring 补上这条前置条件，免得后来者把带旁观者的 group 传进去踩同一个坑。

**离线回归**：`sim_dp_pipeline.py` 现在按通信器记录每个 rank 的入队序列，并断言「集合通信器上各 rank 顺序一致 / P2P 对通信器两侧互为镜像」。修复后 executor 组通信器上只有 `all_gather_object`（四 rank 一致），边界在 `p2p (0,2)` 上镜像；把 P2P 按修复前的归属回放，该断言如期报 MISMATCH——证明这条检查确实承重。

**诊断设施**（测试侧，无生产改动、未放宽超时、未 skip）：每个 worker 用 `faulthandler.dump_traceback_later` 定时把自己的 Python 栈写到 `stack_<rank>.txt`；`_spawn` 超时失败时把各 rank 的栈一并打进报错。下次若再挂，报错本身就会指明每个 rank 卡在哪个 distributed 操作。

本机验证：全量 **81 passed, 8 skipped**；三个离线仿真全过；入口正常。目标机需重跑 `python3 -m pytest -q tests/test_recovery.py` 复核 27 项。

---

## T15：两两组合测试

状态：测试编写完成，**分布式门禁（Gloo + 真实 GPU/NCCL）待 torch 环境执行**（本机无 torch，计划硬性禁止安装/升级 torch）。**未新增任何功能，也未改动 `resihp/` 下任何文件**——本轮暂未发现需要修复的实现缺陷。

新增：`tests/test_combinations.py`（唯一新增文件）。

### 七项组合各自补的是哪条缝

计划四.D 要求两两组合「须执行真实前反向，不只比计划」。逐条核对已有覆盖后，每项都选了**现有门禁没锁住的那条缝**，而不是把已过的场景再跑一遍：

| 组合 | 已有覆盖 | 本轮补的缝 |
|---|---|---|
| TP+PP | T10 是单 stage 的 TP，T12 是 TP degree 1 的流水 | **TP2 跑在 1F1B 里**：ranks (0,1)=stage0 TP 组、(2,3)=stage1 TP 组；激活在 TP 组内是复制的，所以流水 hop 按 TP 下标分两列 (0,2)/(1,3)。每个 rank 的前反向本身就是真 TP all-reduce，且发出的原语序列逐字等于 1F1B |
| TP+DP | T13 的 `tp_heterogeneous` 只手工喂 `dp_combine_gradients` | 整个 `DataParallelRuntime` 驱动合并，合并看到的是**它自己分片前反向产出的** shard |
| PP+DP | T13 有「均衡+跨 replica 重路由」与「不均衡但无流水」 | 两者合起来的形态：**流水化 replica 之间 3-vs-1 的不均衡切分** |
| Scheduler+建群 | T14 走这条路但**从不校验数值** | `build_plan` → `build_training_groups` → 在这些组上训练一步，**逐参与参考比对** |
| Scheduler+状态迁移 | T14 只断言原则 A 的**恢复前**半段 | 补**恢复后**半段：迁移后的下一轮 == 「同 checkpoint 起点 + 新拓扑 + 新配置实际 batch」的参考。此场景两个 replica 迁移后 TP degree 一个 1 一个 2 |
| 状态迁移+checkpoint | T14 断言各 rank 分片等于 checkpoint 切片 | 补**往返**：分片态 gather 成的 anchor 本身就是完整逻辑模型、且等于参考（含 AdamW 矩与 step），各 rank 再从它取回自己那片（无 peer replica，走 checkpoint 兜底分支），然后继续训练仍对 |
| 动态通信组+PipelineRuntime | 无 | 迭代**之间**销毁并重建全部训练组，`PipelineRuntime` 在新通信器上跑第二轮，AdamW 矩状态跨重建带过去，第二轮仍等于参考第二轮。不掺任何故障，被测的只有建组/毁组本身 |

每项都有 Gloo 与 **NCCL** 两个门禁（真实 GPU 张量 + 真实 NCCL 训练组，GPU 不够才 skip）；world 组一律 Gloo，因为它是控制面的常驻组（计划 3.2）。

### 本机能做到的验证（无 torch）

1. **计划形态已实机核对**：`resihp/plan.py` 与 planner 均不依赖 torch，直接跑 `build_plan` 确认了断言里写死的每个数字——`tp_dp` v0/v1 的 micro-batch 划分都是 `[(0,0),(0,1),(1,2),(1,3)]`；v1 的 degree 是 `[1, None, 2, 2]`；`pipeline` v1 的层区间是 stage0 `[0,1]` / stage1 `[2,3,4,5]`（确实既降 degree 又跨 stage 搬了一层）；`solo_pp` 两个 stage 各 2 层。
2. **排队顺序与死锁离线仿真**（沿用 T14 的做法，scratch 脚本不入库）：把每个 rank 建模成阻塞算子序列（集合通信按通信器成员对齐、批量 P2P 按镜像配对），跑会合式仿真。三个新并行场景 `tp_pp` / `tp_dp` / `pp_dp` 全部跑完、无死锁，且每个集合通信器上各成员入队序列一致。
   - **反证 1**：把 `tp_pp` 里融合的 `send_forward+recv_backward` 拆成两次独立收发，仿真如期在第 11 个算子上四个 rank 同时卡在 send——正是 T12 docstring 论证过的确定性死锁，说明这个仿真承重。
   - **反证 2**：把 `pp_dp` 的 stage 边界 hop 改回**批量** P2P、挂到 executor 集合通信器上（即 T13 那个已被 T14 修掉的故障），顺序检查如期报 `exec: {0:7, 1:7, 2:3, 3:3}`。本轮写法沿用修复后的非批量 `isend`/`irecv`，边界拿到的是只含两个 peer 的通信器，故一致。
3. `python3 -m pytest -q`：**81 passed, 9 skipped**（新模块与其它分布式模块一样，因缺 torch 在收集阶段整模块 skip；此前是 8 skipped）。

### 目标机需要复核的点

- `python3 -m pytest -q tests/test_combinations.py` 共 14 项（7 组合 × Gloo/NCCL）。NCCL 侧 6 项需 4 GPU、1 项需 2 GPU。
- 唯一对容差敏感的断言是 `dynamic_groups_pipeline` 的**第二轮**：它从第一轮自己的输出出发，与参考第二轮比，比其它门禁多累一轮重结合误差。若只有它擦边，是容差问题不是缺陷；其余门禁都只跨一轮。

遗留问题：分布式门禁本机无法执行，需在目标机跑完 14 项后才能把状态改为「已验证」。

### 自审修正（提交前）

**修掉一个真 bug（`_stage` 的 layout 与 source 不匹配）**：`shard_dims(layer_ids)` **无条件**含 embedding / final_norm / lm_head 五个边界名（它们总归属某个 stage），而 `shard_logical_state` 会对 layout 里每个名字做 `full_state[name]`。首次建 stage 传的是完整逻辑态，取得到；但 `dynamic_groups_pipeline` 重建 stage 时传的是**该 stage 自己的分片**（stage0 没有 `final_norm`/`lm_head`，stage1 没有 embedding），于是 `KeyError`——两个 rank 每次必崩。改法是让 layout 跟随 source（`if name in source`），而不是给缺失名兜个默认值：`_stage` 的契约本来就是「按你交给它的状态建」，`TensorParallelStage` 再按 `is_first`/`is_last` 各取所需。首次建 stage 的行为逐字不变（完整态与 `shard_dims` 全集相交仍是全集），其余六个门禁不受影响。

**两处 NCCL 断言由「意图」改成「事实」**：`scheduler_migration` / `migration_checkpoint` 原本记 `is_cuda = device.type == "cuda"`——那只是复述自己传进去的参数，即使张量其实落在 CPU 也照样通过。改为在 `_attach` 之后取 `next(run.stage.parameters()).is_cuda`，与其余五个门禁口径一致。

**两处可读性**：`tp_dp` 的 executor 由 `(micro // 2 * 2, micro // 2 * 2 + 1)` 改为直白的 `(0, 1) if micro < 2 else (2, 3)`；`_assert_dynamic_groups_pipeline` 里名为 `rounds` 实为「按 rank 排列」的列表改名 `first_stage` / `last_stage`。

**复核过、确认无误的点**：`ReferenceTransformer.logical_state_dict()` 返回的是真 `nn.Parameter`（不是副本），所以 `_reference_from_anchor` 里 `optimizer.state[named[name]]` 的键能对上、`param.copy_` 也确实写进模型；`PipelineRuntime` 内部没有任何「一个 stage 一个 rank」的隐含假设（激活缓冲用的是完整 `stage.dim`，loss 用的是 all-gather 后的完整 logits），所以 TP2 按列接入是合法用法；异 degree/异 replica 的梯度合并按「每 replica 重建完整张量 → 求和 → 按各自布局重切」逐名走通。

**已知残留（不改，仅记录）**：`PipelineRuntime` 的 docstring 把 `stage_ranks` 描述为「各 stage 的全局 rank」，在 TP>1 时正确用法是**每个 TP 下标一条流水线列**（本轮 `tp_pp` 门禁确立的用法），照字面把某 stage 的全部 TP 成员都传进去会出错。属于文档措辞，不是缺陷，且改 `resihp/` 超出 T15「只补测试与必要修复」的范围，留待需要时再补。
