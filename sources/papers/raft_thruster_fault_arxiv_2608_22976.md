# RAFT（特权 Critic 的无传感器推进器故障适应）

> 来源归档（ingest）

- **标题：** Privileged Critic Training Enables Sensor-Free Thruster Fault Adaptation in End-to-End RL
- **类型：** paper
- **原始链接：**
  - <https://arxiv.org/abs/2608.22976>
- **代码：** <https://github.com/snt-spacer/RAFT>
- **机构：** 卢森堡大学（University of Luxembourg）
- **入库日期：** 2026-08-26
- **一句话说明：** PPO 价值函数在训练时看见真实退化向量 \(D_{gt}\)，actor 始终只看任务观测；部署无需故障传感器即可补偿连续退化、死推进器与卡开阀门。

## 核心摘录（MVP）

### 1) 特权放在 critic，而不是部署观测

- **摘录要点：** 传统 FDI + 切换控制器依赖部署期专用传感器；Oracle 把 \(D_{gt}\) 喂给 actor 不现实。RAFT 问：若价值函数训练时能看见故障，actor 部署还需要故障信息吗？答案是 critic 特权是主机制。
- **对 wiki 的映射：**
  - [RAFT](../../wiki/entities/paper-raft-thruster-fault.md)
  - [Privileged Training](../../wiki/concepts/privileged-training.md) — 非对称 AC，无 teacher 蒸馏。

### 2) 三模式故障 + GRU-64 actor

- **摘录要点：** 8 推进器 + 1 反作用轮浮动平台；Go-to-Position（3 m 半径→5 cm 保持 50 步）。\(u_i^{applied}=s_i u_i+\delta_i\)：DEG 连续缩放、DEAD \(s_i=0\)、STK 常开偏置。课程 \(k_{max}:0\to4\)。Actor 观测 \(\mathbb{R}^{15}\)；critic 额外 16 维 \(D_{gt}\)。
- **对 wiki 的映射：**
  - [RAFT](../../wiki/entities/paper-raft-thruster-fault.md) — 故障模型与架构。
  - [PPO](../../wiki/methods/ppo.md)

### 3) 数字：70.2% 与 84% gap closure

- **摘录要点：** \(k=4\) 混合故障：VAN 4.8%、RAFT **70.2%**、Oracle 82.4%（弥合 84%）。VAN-MLP-AC（无记忆、同 critic）已 66.4%（79% gap）；GRU 再 +3.8 pp。无特权 critic 的 GRU/LSTM 最高 4.0%。OBS-MSE 可读故障估计，代价 −11 pp SR。难度 DEG < STK < DEAD。
- **对 wiki 的映射：**
  - [RAFT](../../wiki/entities/paper-raft-thruster-fault.md) — 评测与消融。

### 4) 开源状态（截至 2026-08-26）

- **摘录要点：** **已开源**。`snt-spacer/RAFT` 提供 Docker、`scripts/rsl_rl/train.py`、reset-time / mid-episode eval，以及 README 所述 checkpoint 布局（Isaac Lab + rsl_rl fork 并排）。
- **对 wiki 的映射：**
  - [仓库归档](../repos/raft_snt_spacer.md)

## 当前提炼状态

- [x] arXiv HTML + GitHub README 已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-raft-thruster-fault.md` 新建
