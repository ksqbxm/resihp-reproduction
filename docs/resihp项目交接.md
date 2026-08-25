# ResiHP 复现项目 — 交接文档

## 项目概述

### 我们在解决什么问题？

大模型训练跑在几十上百张卡上，**单卡挂掉是常态而不是意外**。传统做法是整个作业挂掉、从最近
checkpoint 全量重启：拓扑不变、坏卡换新卡，等不到新卡就干等着。

ResiHP 的思路是：**不换卡，改拓扑**。一张卡永久下线后，用剩下的健康卡重新规划 TP/PP/DP
三个并行维度，把失效卡上的模型分片与优化器状态迁移到幸存卡上，从原进度继续训练。

```
8 卡 TP2×PP2×DP2 正常训练
  │
  ▼ rank1 fail-stop
剩 7 卡：TP 降 degree → PP 重分层 → DP 重路由
  │
  ▼ 状态迁移（param / exp_avg / exp_avg_sq / step）
7 卡新拓扑，从第 3 轮继续训练
```

本仓库是这条**纠错恢复路径**的原生 PyTorch 复现：一个可真实训练的微型 decoder-only
Transformer，加上完整的重规划 + 通信组重建 + 状态迁移链路，支持反复注入故障直到资源耗尽。

### 范围边界（硬性，写在计划文档里）

| 做 | 不做 |
|---|---|
| 纠错恢复：重规划 → 建群 → 状态迁移 → 继续训练 | Detector、心跳、硬件测速、fail-slow、时间预测、性能优化 |
| fail-stop：失效 rank 的进程被真正 `SIGKILL` | 多卡同轮批量失效、跨轮迁移/抢占 |
| 原生 PyTorch + NCCL/Gloo | Megatron；安装/升级/修改 Torch、CUDA、NCCL |
| FP32、固定随机 token、AdamW | 混合精度、多优化器 |

所有存活设备一律视为健康，**pᵢ 恒等于 1**，论文里的速度启发式在本项目中不存在（有专门的门禁扫描全仓，确保
`Detector` / `p_i` / `speed` / `standby` / `Algorithm 1` 这些构造命中数为 0）。

### 两条根本原则（贯穿全文）

**原则 A — 数值验收两段式**

```
恢复前：完整逻辑参数 + AdamW 状态 与故障前 checkpoint 逐张量精确相等（torch.equal）
恢复后：与「同一 checkpoint 起点 + 新拓扑 + 新配置实际 batch + 同种子」重建的参考运行逐步一致
```

注意后半句：参考基准是**新配置从同起点**，绝不是「无故障从头跑」。踢掉一个 rank 后 batch 划分与归约结构都变了，
和无故障参考必然分岔——那是语义不同，不是精度误差。

**原则 B — 纯 fail-stop 确定性重路由**

DP 数据重路由是**纯函数**：`(step, failure_signature, active_topology) → assignment`。
同输入必同输出，不读运行时延迟、不看进度、不做模拟到达时间。`failure_signature` 由「到当前事件为止累计失效的 rank 集合」唯一确定。

---

## 第一章：环境搭建

### 1.1 运行环境

CPU/Gloo 测试和 GPU/NCCL 测试**在同一个容器里跑**，共用同一套 Scheduler、状态格式和执行路径，只切后端。

| 项目 | 配置 |
|---|---|
| 宿主机 | Ubuntu Linux |
| Docker 容器名 | `resihp` |
| Docker 镜像 | `resihp-env:gpu-fix-backup` |
| 容器系统 | Ubuntu 24.04 |
| Python | 3.12 |
| PyTorch | 2.8.0a0+5228986（NVIDIA PyTorch 25.06） |
| CUDA / NCCL | 12.9.1 / 2.27.3 |
| GPU | 容器开放全部 8 张 |
| 宿主机路径 | `/data/ubuntu/resihp-project` |
| 容器内路径 | `/workspace/resihp` |

镜像里已包含 PyTorch、CUDA、NCCL，**不需要单独安装 CUDA 或编译 NCCL**。

### 1.2 进入容器

代码放在服务器 `/data/ubuntu/resihp-project`，通过 bind mount 映射进容器：

```
/data/ubuntu/resihp-project   ──bind mount──▶   容器 resihp:/workspace/resihp
```

两边看到的是同一份代码，宿主机 `git pull` 后容器内立刻可见，不需要再拷贝。

```bash
docker start resihp && docker exec -it resihp bash
```

进去以后 `cd /workspace/resihp`。退出用 `exit`，停容器用 `docker stop resihp`。

### 1.3 容器重建（容器被删了才需要）

```bash
docker run --gpus all -itd \
  --name resihp \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /data/ubuntu/resihp-project:/workspace/resihp \
  -w /workspace/resihp \
  resihp-env:gpu-fix-backup \
  bash
```

| 参数 | 作用 |
|---|---|
| `--gpus all` | 开放全部 GPU |
| `--ipc=host` | 用宿主机 IPC，多进程通信不受共享内存限制 |
| `--ulimit memlock=-1` | 解除 locked memory 限制 |
| `--ulimit stack=67108864` | 进程栈 64 MB |
| `-v ...:/workspace/resihp` | 挂载项目代码 |
| `-w /workspace/resihp` | 默认工作目录设为项目根 |

换新服务器时先确认 `resihp-env:gpu-fix-backup` 镜像已导入，再执行上面这条。

### 1.4 环境检查

进容器后先确认 PyTorch 认得 GPU、两个后端都在：

```bash
python - <<'PY'
import torch
import torch.distributed as dist
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "| GPU count:", torch.cuda.device_count())
print("NCCL available:", dist.is_nccl_available(), "| Gloo available:", dist.is_gloo_available())
print("NCCL version:", torch.cuda.nccl.version())
PY
```

正常应为 `CUDA available: True` / `GPU count: 8`。本项目同时用两个后端：**Gloo** 跑 CPU 测试和每个 epoch 的 world 组，
**NCCL** 跑真实 GPU 训练通信（TP / PP / DP 三类组）。

跑多卡任务前先 `nvidia-smi` 确认目标卡空闲，或用 `CUDA_VISIBLE_DEVICES` 指定——历史上「NCCL 卡死」的真实原因是显存被别的进程占了。

### 1.5 获取和更新代码

仓库 `ksqbxm/resihp-reproduction`，当前开发分支 `fix/dp-replica-level-rerouting`。

```bash
cd /data/ubuntu/resihp-project && git checkout fix/dp-replica-level-rerouting && git pull origin fix/dp-replica-level-rerouting
```

### 1.6 快速验证

```bash
python -m pytest -q
```

只想先确认基础环境能跑：

```bash
python -m pytest -q tests/test_reference.py
```

---

## 第二章：代码结构

```
resihp/
├── config.py      配置与故障文件校验（唯一入口的守卫）
├── memory.py      解析显存模型（全项目唯一计算器）
├── model.py       自建 decoder-only Transformer，稳定全局 layer id
├── reference.py   单进程确定性参考训练（所有数值比对的锚点）
├── checkpoint.py  原子 checkpoint 保存 / 恢复
├── plan.py        不可变、带版本号的 ExecutionPlan + 不变量断言
├── planner/       纯函数规划层（不碰通信）
│   ├── tp.py      TP 候选生成与确定性成员选择
│   ├── pp.py      PP 重分层 + 1F1B 调度定义
│   └── dp.py      DP 确定性重路由
├── parallel/      真实执行层
│   ├── tp.py      TensorParallelStage：真分片前反向（f/g 两个 Megatron 集合）
│   ├── pp.py      PipelineRuntime：全项目唯一运行时，1F1B 调度
│   ├── dp.py      跨 replica 路由与梯度合并
│   └── reshard.py TP 重切（donor 收集 / checkpoint 兜底 / 重 chunk）
├── membership.py  store 协议 + Supervisor：托管 TCPStore、收尸、公布每轮成员表
├── control.py     控制面、安全点、world 重建、plan/torch rank 翻译、一致停止
├── recovery.py    唯一恢复链路（执行 plan 的 state_routes）
├── verify.py      原则 A 的两个契约，唯一定义
├── train.py       worker：一个 rank 的训练控制循环（含 SIGKILL 注入）
└── launch.py      唯一入口：起 worker、托管 store、子进程死掉时不动其余进程
```

**分层纪律**：`planner/` 是纯函数（同输入必同输出、可单进程穷举测试），`parallel/` 才碰通信。
通信 rank / TP shard / PP owner **只由当前 ExecutionPlan 决定**，禁止从旧布局隐式推导。

---

## 第三章：功能全景

### 功能一：确定性训练基线与原子 checkpoint

**背景**：整个项目的验收都靠「和参考一致」，所以必须先有一个字节级可重复的锚点。

模型是手搭的（token/position embedding、LayerNorm、多头因果自注意力、MLP、残差、LM head），
不用现成的 Transformer 模块——因为**每个 layer 需要稳定的全局 ID、每个参数需要稳定的逻辑名**
（如 `layers.2.attn.q_proj.weight`）。层搬到别的 PP stage、按别的 TP degree 重切之后名字不变，才定位得到。

checkpoint 的原子性（计划 3.6）：先写临时文件 → 完整读回校验 → `os.replace` 覆盖。
任何失败都删掉临时文件、保留上一份有效 checkpoint。全项目只有**一个** checkpoint 文件。

持久恢复状态的唯一定义：

```
param / exp_avg / exp_avg_sq / step / iteration（= 数据游标） / RNG
```

**`grad` 不在其中**，这是刻意的：安全点只出现在「当前迭代完成 → AdamW step 完成」之后，且不存在跨安全点的梯度累积，
下一轮必然先 `zero_grad` 再重算。所以安全点时刻的 `param.grad` 是**已经被消费掉的旧值**，搬它没有任何语义价值。

```bash
python -m pytest -q tests/test_reference.py tests/test_checkpoint.py
```

### 功能二：ExecutionPlan 与九步安全点

**背景**：三个并行维度同时变，必须有一个所有 rank 都认同的、不可变的真值来源，否则各 rank 各自推导必然打架。

`ExecutionPlan` 是 `(config, step, version, 累计失效 rank 集合, 上一版计划)` 的确定性函数，含：
活跃/失效 ranks、每个 DP replica 每个 PP stage 的 TP 成员、每 stage 的连续 layer 区间、
每个 micro-batch·stage 的执行 rank、状态迁移路由（`state_routes`）、计划版本与规范化 digest。

**故障是真的 kill**：被点名的 rank 在完成本轮、写完 checkpoint 之后 `SIGKILL` 自己，
进程当场消失，没有清理、没有告别通信、显存和通信域随进程一起没了。由此带来三件事：

| 组 | 后端 | 生命周期 |
|---|---|---|
| world 组 | Gloo | 一个 epoch 一个：控制类集合通信（停止协商、checkpoint 收集、恢复收集）跑在它上面。含死进程的那个**永久不可用**（`new_group` 是它上面的集合操作），所以每次 fail-stop 整体销毁、在幸存者上重建 |
| 训练组 | GPU=NCCL / CPU=Gloo | 每次故障全体同步销毁重建：每 stage 一个 TP 组、每个流水跳一个并集组、一个覆盖全部在岗 rank 的 DP 组 |
| store | `TCPStore` | 托管在 **launcher**（不是 rank 0）里，全程存活：它是唯一一条死进程堵不住的通道 |

**谁发现故障**：launcher。它是起进程的父进程，唯一依据是操作系统报告的子进程退出状态——
没有心跳、没有超时猜测、不可能误判活着的 rank。每轮迭代边界上所有存活 rank 在 store 上会合
（`ControlPlane.observe`），launcher 收完尸再公布本轮成员表；被杀的 rank 从来没到过会合点，
所以幸存者是在**发起下一次训练集合通信之前**就知道它没了——NCCL 永远不会拿到死对端。

**plan rank 与 torch rank**：plan rank 是进程启动时的固定身份（planner、checkpoint、
`ExecutionPlan` 只讲这一种）；torch rank 是它在当前成员表里的下标，每次重建都变。翻译只在
`ControlPlane.torch_rank` 一处发生；`PipelineRuntime` 的 P2P 对端则走 hop 组的组内下标换算。

安全点（`ControlPlane.safe_point`）：

```
1 完成当前迭代           2 原子保存 checkpoint（每轮都做）  3 边界会合，读回成员表
4 标记该 rank 永久失效   5 TP→PP→DP 重规划                  6 释放旧组与旧 world、
                                                              在幸存者上重建 world、建新训练组
7 恢复/迁移/重切状态     8 校验计划与状态摘要一致            9 从下一迭代继续
```

第 2 步为什么每轮都做：死进程事后没法贡献任何分片，唯一还能持有它那份的 checkpoint，
就是它**活着时**写下的那个——「恢复前状态精确等于故障前 checkpoint」这条验收才有意义。

```bash
python -m pytest -q tests/test_plan.py tests/test_control.py
```

### 功能三：TP 重规划与真实张量重切

**背景**：TP 组里死了一张卡，剩下的成员凑不出原来的 degree，必须换 degree 并把每个逻辑张量按新布局重新切。

候选 degree：`K = { k | k_min ≤ k ≤ |G'|, k = 2^q }`，且要满足

- attention heads 与 MLP 分片维度能被 k 整除；
- **vocab size 能被 k 整除**（embedding / LM head 是 vocab-parallel，漏了这条 planner 会发布一个 runtime 建不出来的布局）；
- 解析显存模型容纳得下。

因为 pᵢ≡1，论文的 `argmax k·min(pᵢ)` 退化为：**取最大可行 degree，成员按 rank 升序**。没有 pᵢ 评分分支，没有多套 fallback。

重切的四步（计划 3.3）：

```
1. 从其他健康 DP replica 收集完整逻辑张量
2. 某 shard 在所有健康 replica 都缺失时 → 从故障前 checkpoint 恢复（唯一兜底）
3. 按新 degree 重 chunk param / exp_avg / exp_avg_sq
4. 分发到新 TP 组，gather 后与 checkpoint 逐张量校验
```

**异构 TP 边界**（上游 stage TP1、下游 stage TP2 这种）：stage 的激活在自己 TP 组内是复制的，
所以边界只搬**一份权威副本**——两个 stage 的 leader 之间点对点，接收方在自己 TP 组内 broadcast。
既不是 per-rank 求和（会把梯度翻倍），也不是只给一个 rank（其他 rank 会饿死）。全项目只有这一套实现。

```bash
python -m pytest -q tests/test_planner_tp.py tests/test_parallel_tp.py tests/test_parallel_reshard.py
```

### 功能四：PP 重分层与 1F1B 运行时

**背景**：TP degree 变了以后，各 stage 的容量比例就变了，层数得跟着重新分。

`new_tp_degrees` 是每个既有 stage 的**新 TP degree**，不是目标层数。目标层数
`L_target = floor(L_old × TP_new / TP_old)`；有 TP 组的 stage 至少 1 层，TP 组全灭的 stage 为 0 层（stage 被清空）。
然后调整到守恒原模型总层数：正差值逐层给「层数/TP degree」最小者，平局按 stage ID 升序；负差值从比值最大者收回，平局按 stage ID 降序。
比例比较用整数交叉乘法，不引入浮点。返回前强制验证：每 stage 是连续区间、所有全局 layer 恰出现一次、总层数守恒。

**1F1B 调度定义在 `planner/pp.py` 而不是运行时里**，因为有三个调用方必须对它达成一致，而其中只有一个能 import torch：
`PipelineRuntime` 执行它，`plan.py` 和 `planner/dp.py` 靠它算峰值激活显存。
`peak_in_flight` 是**回放调度 +1/−1 得出的**，不是闭式常量——调度改了，显存预算自动跟着改。

`PipelineRuntime` 是全项目**唯一**运行时：控制面驱动它，测试也直接测它，不存在「测试专用调度器」。
它执行 warmup 前向 → 1F1B → cooldown 反向，micro-batch 梯度累积成一次 AdamW 更新。

```bash
python -m pytest -q tests/test_planner_pp.py tests/test_parallel_pp.py
```

### 功能五：DP 确定性重路由

**背景**：一个 replica 的 stage 被打空了，它原本要跑的 micro-batch 得交给别的 replica。

先生成新拓扑，再按新拓扑的可达容量做**容量比例静态划分**。映射是纯函数，对同输入幂等。
迁移前用**同一个**显存计算器判 `MemoryFeasible`，不可行就按确定顺序试下一候选，全不可行则判计划不可行。

必须保证的几条：每个 micro-batch 每个 stage 恰好执行一次；前向 activation 发给**实际的**下游 executor、
反向 gradient 回到**实际的**上游 executor；executor 换人不影响 global batch 的梯度归一化；
不同 replica 的 PP 分层与 TP degree 可以不一样，梯度照样合得起来（在**完整逻辑张量**上求和后再按各自布局重 chunk）。

```bash
python -m pytest -q tests/test_planner_dp.py tests/test_parallel_dp.py
```

### 功能六：唯一恢复链路与一致停止

**恢复链路**（计划 3.6 只允许这一条）：

```
fail-stop → 读 checkpoint / 从健康 replica 收集 → 恢复完整逻辑状态 → 重排计划 → 建群 → 继续
```

`recovery.py` 是**纯编排**，而且编排的是计划已经决定好的事：planner 发 `StateRoute`
（哪个 state group、从哪个 donor、源布局是什么、目标布局是什么），`recover()` 只负责执行。
运行时不再有第二套路由策略，只保留计划做不了的安全检查：donor 集合真不完整才落 checkpoint、
route 声明的名字必须真的到货、stage 该有的名字必须被覆盖——三者都抛错走一致停止。

**一致停止**：六条停止条件，任何一条被任何幸存 rank 观察到，**所有幸存 rank 抛同一个
`ConsistentStop`**。协商跑在**刚重建好的 world 组**上（旧的那个含死进程，一次集合通信都跑不了），
原因取自 gather 到的列表而不是本地视角，而且发生在**建任何训练组之前**。所以不会有人半路继续、
不会留半完成的计划或组，故障前 checkpoint 原封不动。

其中 `no_executable_pp`（所有 replica 全灭）在真 kill 下等于一个进程都不剩：没有 rank 能观察它、
协商它、正常退出。它因此不再是 rank 观察到的停止，而是作业结束本身——launcher 汇总退出码，
最后一次完成迭代的 checkpoint 留在盘上可重载。planner 侧这条不可行原因仍由纯函数测试锁定。

| 停止码 | 含义 |
|---|---|
| `no_feasible_tp` | 没有可行 TP degree |
| `no_executable_pp` | 没有可执行的 PP 分区 |
| `no_feasible_dp_target` | DP 找不到满足显存的目标 |
| `checkpoint_unusable` | checkpoint 缺失/损坏/摘要不匹配 |
| `plan_disagreement` | 各 rank 计划摘要不一致 |
| `state_mismatch` | 状态重切后完整逻辑张量对不上 |

```bash
python -m pytest -q tests/test_recovery.py
```

### 功能七：解析显存模型（唯一计算器）

`resihp/memory.py` 是**全项目唯一**的显存公式：分片参数 + 边界参数 + 梯度 + AdamW 两份矩状态 + 峰值 activation。
TP 的 `k_min` 搜索和 DP 的 `MemoryFeasible` 门都调它，不存在第二份公式。

两个刻意的设计：

- `in_flight_micro_batches` **没有默认值**。默认 1 会静默低估每个 warmup 阶段的激活峰值，
  所以调用方必须显式从 `peak_in_flight` 取值。
- 激活只计入「尚未完成对应 backward」的 micro-batch，不是把全部 micro-batch 线性相加。
  TP 组内的中间激活除以 `tp_degree`，all-reduce 之后的 stage 输入/输出不除（每个 TP rank 都是完整份）。

它读的是配置不是硬件——**本项目任何地方都不读 GPU 实际显存**。`memory_budget_bytes` 留 `null` 就是不设人为上限。

```bash
python -m pytest -q tests/test_memory.py
```

---

## 第四章：使用方法

### 4.1 唯一入口

```bash
python3 -m resihp.launch --config configs/train.json --failures configs/failures.json
```

入口不是 `torchrun`：elastic agent 见到一个 worker 被信号杀死就会连带杀掉/重启其余 worker，
而这里要的是幸存者原地重配。`resihp.launch` 托管 store、按 world size 起 worker、子进程死掉时
不动其余进程，并汇总每个进程的退出码（被杀的应为 -9）。`resihp.train` 是它起的 **worker**。

`resihp.train` 不带 `RANK` 环境变量直接跑，它退化成一个不依赖 torch 的配置回显，
方便单独检查 CLI 和配置文件：

```bash
python -m resihp.train --config configs/train.json --failures configs/failures.json
```

### 4.2 `configs/train.json`

字段是**封闭集合**：少一个报缺失，多一个报未知字段，不做任何默认值兜底。

```json
{
  "model_dim": 128, "num_layers": 6, "num_heads": 8,
  "batch_size": 8, "micro_batch_size": 2, "seed": 1234,
  "tp": 2, "pp": 2, "dp": 2, "iterations": 8,
  "memory_budget_bytes": null
}
```

约束：`model_dim % num_heads == 0`、`batch_size % micro_batch_size == 0`、world size = `tp × pp × dp`。
`memory_budget_bytes` 是唯一可选字段（`null` = 不设上限），也是在真实运行里打开显存门禁的唯一开关。

`VOCAB_SIZE = 256` 和 `SEQUENCE_LENGTH = 16` 是**运行常量**而不是配置字段，写在 `resihp/train.py` 里：
配置 schema 只认计划列出的那些字段，而这两个值固定 token 流和 embedding 形状。
两者都会传给 planner——vocab 决定哪些 TP degree 能整除，seq len 决定激活预算。

### 4.3 `configs/failures.json`

```json
{"events": [
  {"after_iteration": 2, "failed_rank": 1},
  {"after_iteration": 3, "failed_rank": 4}
]}
```

只允许这两个字段；`after_iteration` 严格递增；`failed_rank` 在 world size 内且不重复。
**出现速度/降级/检测类字段直接拒绝并报出字段名**——这是守卫，不是实现。
每个 event 恰好使一个新 rank 永久失效，独立触发一次完整重规划与恢复。

### 4.4 怎么读输出

跑完的 run，每个 rank 打印一行：

```json
{"acceptance": {"rank": 0, "device": "cuda", "training_backend": "nccl",
  "trained_iterations": [1,2,3,4,5,6,7,8], "plan_versions": [0,1,2,3,4,5],
  "plan_digests": ["..."], "failed_ranks": [1,4,5,6,7]}}
```

一致停止的 run 打印的是：

```json
{"stopped": "no_executable_pp", "reason": "..."}
```

退出码本身证明不了「故障真的发生过而且训练继续了」，这行 JSON 才是可从外部验证的证据。

### 4.5 默认配置的计划演化

默认配置排了**五次连续 fail-stop**，故意让三个维度都真的动起来（层数取 6 而不是 4 是有原因的：
4 层时重分层恰好把原层数还给每个 stage，就只看得到 TP 变化）：

| 版本 | 生效轮次 | 本次失效 rank | replica 0 | replica 1 | micro-batch 归属 |
|---|---|---|---|---|---|
| v0 | 1–2 | — | s0 TP2(0,1) L[0-2] / s1 TP2(2,3) L[3-5] | s0 TP2(4,5) L[0-2] / s1 TP2(6,7) L[3-5] | r0=[0,1] r1=[2,3] |
| v1 | 3 | 1 | s0 **TP1**(0) **L[0-1]** / s1 TP2(2,3) **L[2-5]** | 不动 | 不变 |
| v2 | 4 | 4 | 不动 | s0 **TP1**(5) **L[0-1]** / s1 TP2(6,7) **L[2-5]** | 不变 |
| v3 | 5 | 5 | 不动 | **s0 清空**，s1 TP2(6,7) **L[0-5]** | 不变 |
| v4 | 6 | 6 | 不动 | s1 **TP1**(7) L[0-5] | 不变 |
| v5 | 7–8 | 7 | 不动 | **整个 replica 消失** | **r0=[0,1,2,3]** |

TP 在 v1/v2/v4 降 degree，PP 在 v1/v2 搬层、v3 清空 stage，DP 在 v5 真正重路由 micro-batch。
各 rank 实际执行的轮次：rank 1 = `[1,2]`、rank 4 = `[1,2,3]`、rank 5 = `[1..4]`、rank 6 = `[1..5]`、
rank 7 = `[1..6]`、rank 0/2/3 = `[1..8]`。

---

## 第五章：测试与验收

### 5.1 测试地图

| 文件 | 覆盖 | 需要 |
|---|---|---|
| `test_config.py` / `test_memory.py` | 配置校验、显存公式（含手算逐项核对与「刚好满足 / 超一字节」边界） | 无 |
| `test_planner_{tp,pp,dp}.py` / `test_plan.py` | 三个 planner 纯函数 + ExecutionPlan 不变量 | 无 |
| `test_acceptance.py` | 禁用构造全仓扫描 + 默认排程真的动三个维度 + launcher 验收（含被杀进程退出码） | 后者需 8 GPU |
| `test_entrypoint.py` | 唯一入口可解析 | 无 |
| `test_reference.py` / `test_checkpoint.py` | 参考训练可重复、checkpoint 原子性 | torch |
| `test_parallel_{tp,pp,dp,reshard}.py` | 真实分片前反向、1F1B、跨 replica、TP 重切与异构边界 | torch（Gloo/NCCL 双跑） |
| `test_control.py` / `test_recovery.py` | 安全点与真 kill、恢复路径、五条 rank 可观察的一致停止 + 全灭即作业结束 | torch，8 进程 |
| `harness.py` | 多进程门禁共用的 supervisor（托管 store、收尸、公布成员表），不是测试 | torch |
| `test_combinations.py` | 七项两两组合，全部执行真实前反向 | torch，Gloo/NCCL 各一遍 |
| `test_end_to_end.py` | 完整 3D `TP2×PP2×DP2` 八进程端到端 | torch，Gloo/NCCL 各一遍 |
| `test_fault_sequences.py` | 固定 seed 随机故障序列、反复注入、donor 耗尽 | torch，Gloo/NCCL 各一遍 |

### 5.2 目标机验收顺序

前一条不过就不用往下走。

```bash
python -m pytest -q tests/test_acceptance.py -v
```

```bash
python3 -m resihp.launch --config configs/train.json --failures configs/failures.json
```

```bash
python -m pytest -q
```

第二条是人工验收：屏幕上应出现 **3 行** `{"acceptance": ...}`（rank 0/2/3，`trained_iterations`
均为 `[1..8]`，三行 `plan_digests` 完全相同、各 6 项），外加一行 `{"launch": ...}`，其中
`killed_ranks == [1,4,5,6,7]`、这五个进程退出码为 -9、其余为 0，且不出现 `{"stopped": ...}`。
被杀的 rank 一个字都不会打印——那正是它们真的死了。
跑之前先清掉 `checkpoint.pt`，跑完它应存在且没有 `checkpoint.pt.tmp` 残留。

### 5.3 当前状态

- **开发机（Windows，无 torch）**：`python -m pytest -q` → **125 passed, 12 skipped**。
  12 项 skip 全是分布式/GPU 模块在收集阶段整模块跳过（`torch` 不可导入），不是被跳过的断言。
- **8 卡目标机**：最近一轮全量审阅改动了真实通信结构（每跳两 rank 组、leader→leader + TP broadcast、
  1F1B 接入生产、recovery 改为执行 `state_routes`），**分布式与数值门禁需要在目标机复跑一遍**。
  旧的目标机确认是针对已经不存在的代码取得的，不能沿用。

逐条自检见 `docs/ACCEPTANCE.md`；实现细节与每一轮修改的根因见 `docs/PROGRESS.md`（项目进度只记在这一个文件）。

---

## 第六章：明确不包含 / 已知边界

| 项 | 状态 | 说明 |
|---|---|---|
| 自动故障检测 | ❌ | 故障来自确定性故障表，不是运行时检测 |
| 多卡同轮失效 | ❌ | 每个事件恰好一个 rank（计划锁定） |
| fail-slow / 速度启发式 | ❌ | pᵢ≡1，相关构造全仓扫描命中为 0 |
| 多节点跨机 | ❌ | 单机 8 卡 |
| 集体通信 mid-flight 恢复 | ❌ | 只在安全点（迭代 + AdamW step 完成后）重配 |
| 跨安全点梯度累积 | ❌ | 一旦引入，`grad` 的恢复契约必须先改 |
| 性能优化 | ❌ | 异构 TP 边界只做功能正确，不做 P2P 优化 |

数值口径上的两点，记录而不掩盖：

- TP all-reduce 和微批切分改变了 FP32 累加顺序，所以除 checkpoint 比对用 `torch.equal` 外，
  训练数值走 `rtol=1e-4 / atol=1e-5` 的重结合带；梯度量级低于 `atol` 的分量不构成逐分量相对验收。
- `resihp/model.py` 模块级钉死 `torch.set_float32_matmul_precision("highest")`：
  TF32 是进程级默认且各 torch 版本变过，计划要求全程 FP32，这条数值契约必须显式声明而不是继承默认。

本项目不宣称证明所有硬件绝对无误；结论建立在确定性数值对照、状态不变量、故障原子性与多进程端到端之上。
