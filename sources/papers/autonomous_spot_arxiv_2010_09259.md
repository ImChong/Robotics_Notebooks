# Autonomous Spot：NeBula 长程地下探索（arXiv:2010.09259）

> 论文来源归档（ingest）

- **标题：** Autonomous Spot: Long-Range Autonomous Exploration of Extreme Environments with Legged Locomotion
- **类型：** paper / quadruped / autonomy / perception / subterranean
- **arXiv：** <https://arxiv.org/abs/2010.09259> · PDF：<https://arxiv.org/pdf/2010.09259>
- **机构：** NASA JPL / Caltech（NeBula 团队；作者含 Amanda Bouman, Ali-akbar Agha-mohammadi, Joel Burdick 等）
- **入库日期：** 2026-07-05
- **一句话说明：** 将 **NeBula（Networked Belief-aware Perceptual Autonomy）** 自主架构与 **Boston Dynamics Spot** 集成，面向 **DARPA Subterranean Challenge** 等极端环境的 **长距离、长时程** 腿足探索；覆盖 mobility、感知、自主决策与组网。

## 核心摘录（面向 wiki 编译）

### 1) NeBula + Spot 系统边界

- **要点：** 论文定位为 **首批在 Spot 上实现大规模长时自主** 的工作之一；强调 **足式机动性** 与 **信念感知自主（belief-aware perception）** 在未知地下环境中的组合，而非仅低层步态控制。
- **对 wiki 的映射：**
  - [`wiki/entities/paper-autonomous-spot-nebula-exploration.md`](../../wiki/entities/paper-autonomous-spot-nebula-exploration.md)
  - [`wiki/entities/boston-dynamics.md`](../../wiki/entities/boston-dynamics.md)

### 2) 能力栈：mobility / perception / autonomy / networking

- **要点：** 分模块讨论 **硬件与软件挑战**：地形机动、感知融合、自主规划与决策，并简要涉及 **无线组网**；在 **真实场景物理系统** 上验证。
- **对 wiki 的映射：**
  - [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)
  - [`wiki/entities/quadruped-robot.md`](../../wiki/entities/quadruped-robot.md)

### 3) 与 BD 原生 AutoWalk 的分工

- **要点：** Spot 商业化栈侧重 **工业巡检 AutoWalk**；NeBula 路线侧重 **极端未知环境探索** 与 **多机器人信念共享**，属于 **研究型自主栈** 而非出厂产品功能。
- **对 wiki 的映射：**
  - [`wiki/entities/paper-autonomous-spot-nebula-exploration.md`](../../wiki/entities/paper-autonomous-spot-nebula-exploration.md)

## 当前提炼状态

- [x] 摘要级摘录与 wiki 映射
- [ ] 与 SubT 决赛技术报告、CoSTAR 团队后续论文交叉链接
