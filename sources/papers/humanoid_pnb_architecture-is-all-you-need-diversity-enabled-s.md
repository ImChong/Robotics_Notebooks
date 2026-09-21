# Architecture Is All You Need: Diversity-Enabled Sweet Spots for Robust Humanoid Locomotion

> 来源归档（ingest · 深读 · arXiv:2510.14947）

- **标题：** Architecture Is All You Need: Diversity-Enabled Sweet Spots for Robust Humanoid Locomotion
- **作者：** Blake Werner, Lizhi Yang, Aaron D. Ames
- **类型：** paper / humanoid / locomotion / layered-control
- **arXiv：** <https://arxiv.org/abs/2510.14947>
- **入库日期：** 2026-06-11
- **深读更新：** 2026-09-21
- **一句话说明：** 在 Unitree G1 上证明 **分层控制架构（LCA）**——高速本体感觉稳定器 + 低速局部高度图导航——比单阶段端到端感知策略 **更鲁棒**；两阶段课程（先 blind 后感知）是关键，而非网络规模。

## 核心摘录（面向 wiki 编译）

### 1) 分层控制架构（LCA）与「sweet spot」

- **要点：** 快层：关节空间 proprio 跟踪（标准 locomotion RL 奖励）；慢层：11×11 局部高度图 $H^\pi\in\mathbb{R}^{11\times 11}$（1 m×1 m）给出长视野地形参考。性能来自 **多速率信息分工**，而非复杂估计器/混合整数足步/世界模型。
- **对 wiki 的映射：** [`wiki/entities/paper-notebook-architecture-is-all-you-need-diversity-enabled-s.md`](../../wiki/entities/paper-notebook-architecture-is-all-you-need-diversity-enabled-s.md)

### 2) 两阶段训练课程

- **Stage 1（blind）：** actor 侧 $H\equiv 0$，critic 仍见完美感知；四类地形各 25%（上楼梯/下楼梯/ uneven / 平地）。
- **Stage 2（perception）：** 对 actor 重新启用 $H$，在已稳定 blind 策略上学习长视野条件化。
- **对 wiki 的映射：** 同上

### 3) 观测、网络与训练设定

- **要点：** 历史长度 $K$ 的 proprio 栈 + 速度指令 + 上一动作；感知编码器 CNN 或 MLP→$z_H$；actor 为 MLP 或 LSTM（512-256-128）。Isaac Sim，4096 并行 env，非对称 actor-critic，40k steps；7 种变体（blind / 3 one-stage / 3 two-stage）。
- **对 wiki 的映射：** 同上

### 4) 仿真与真机结果（深读）

- **仿真 OOD uneven：** two-stage 相对 one-stage 成功率约 **+10 百分点**，接触次数更低。
- **真机 Table V（G1，5 次试验）：** One-Stage MLP 在楼梯/ledge **0–1/5**；Two-Stage CNN 楼梯 ascent 4/5、descent 5/5、两种 ledge 5/5；Blind 在 ledge 强但楼梯 ascent 仅 3/5。
- **对 wiki 的映射：** 同上

### 5) 感知栈与部署

- **要点：** 双 RealSense D435（髋后 + 胸下视）→ 30 Hz 深度点云；11×11 min-pool + 离群 clamp；策略 50 Hz，关节 PD 1 kHz；感知在机载 Jetson Orin NX，策略在 Framework 笔记本（可背负）。
- **对 wiki 的映射：** 同上

## 开源边界（步骤 2.5）

| 状态 | 说明 |
|------|------|
| **截至 2026-09-21** | arXiv 与项目页 **未列出** 官方 GitHub；按「**未开源**」标注，待作者发布后再建 `sources/repos/` |
| **复现依赖** | Isaac Sim + Unitree G1 + 双深度相机；算法侧核心是 **课程 + LCA 接口** 而非重型感知 |

## 对 wiki 的映射

- [paper-notebook-architecture-is-all-you-need-diversity-enabled-s.md](../../wiki/entities/paper-notebook-architecture-is-all-you-need-diversity-enabled-s.md)
- [paper-notebook-category-05-locomotion.md](../../wiki/overview/paper-notebook-category-05-locomotion.md)
- [humanoid-rl-motion-control-body-system-stack.md](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2510.14947>
- 相关理论：Doyle/Ames 分层控制与 diversity-enabled sweet spots（PNAS 2021）
