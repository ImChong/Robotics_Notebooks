---
type: entity
tags: [humanoid, sim2real, loco-manipulation, teleoperation, visual-rl, reinforcement-learning, cmu, nvidia]
status: complete
updated: 2026-05-14
related:
  - ./gr00t-visual-sim2real.md
  - ./humanoid-robot.md
  - ./zhengyi-luo.md
  - ./xue-bin-peng.md
  - ../concepts/sim2real.md
  - ../tasks/loco-manipulation.md
  - ../tasks/teleoperation.md
  - ../methods/sonic-motion-tracking.md
  - ../concepts/privileged-training.md
sources:
  - ../../sources/sites/tairan-he.md
summary: "何泰然（Tairan He）为 CMU RI 博士、NVIDIA GEAR 实习背景，研究聚焦人形规模化学习与视觉 Sim2Real；代表作含 OmniH2O、HOVER、ASAP、VIRAL / DoorMan 等，主页为论文与项目总索引。"
---

# Tairan He（何泰然）

## 一句话定义

**Tairan He** 是面向 **通用人形 loco-manipulation** 的机器学习研究者：博士阶段在 CMU LECAR 与 NVIDIA GEAR 等合作网络中，系统推进 **特权教师 RL → 视觉学生 / 蒸馏 / 动力学对齐** 的 Sim2Real 管线，并以 [OmniH2O](https://omni.human2humanoid.com/) 等工作把 **全身遥操作接口与数据闭环** 推到社区可见的基准线。

## 为什么重要

- **问题域与仓库主线重合**：其公开论文串覆盖 [Sim2Real](../concepts/sim2real.md)、[Privileged Training](../concepts/privileged-training.md)、[Loco-Manipulation](../tasks/loco-manipulation.md)、[Teleoperation](../tasks/teleoperation.md) 与规模化仿真训练，是理解「2023–2026 人形学习论文潮」的 **作者级索引**。
- **工程可检索**：个人页集中给出项目站与 BibTeX，比零散抓 arXiv 更适合作为 ingest 溯源与后续 curator 补链入口。

## 核心研究脉络（归纳）

1. **接口与数据**：OmniH2O 一类工作强调 **人–机共享运动表示** + 遥操作数据，对应「如何低成本获得全身演示」。
2. **仿真规模化 + 视觉迁移**：VIRAL、DoorMan 等强调 **大规模并行仿真与 RGB 策略**，与仓库中 [GR00T-VisualSim2Real](./gr00t-visual-sim2real.md) 所归纳的 Teacher–Student 叙事同族。
3. **动力学与多智能体分解**：ASAP（残差 / 对齐）、SoFTA、FALCON 等体现 **把全身问题拆成可学子模块** 或 **显式补偿 sim–real gap** 的路线。

## 流程总览（反复出现的 Sim2Real 骨架）

下列图只描述其多篇论文共享的 **逻辑骨架**，细节奖励、观测与算法名以各论文为准。

```mermaid
flowchart LR
  Sim["规模化仿真\n域随机 / 渲染"] --> Teacher["特权教师\n状态观测 RL"]
  Teacher --> Student["学生策略\nRGB / 稀疏传感"]
  Student --> Align["对齐模块\n蒸馏 / DAgger /\n残差动力学"]
  Align --> Real["真机部署\n零样本或少微调"]
```

## 常见误区或局限

- **主页 ≠ 最新事实源**：任职机构、在投论文状态以 **论文页与机构新闻** 为准；本页不替代一手引用。
- **合著与一作分工**：如 SONIC 等以合作者一作为主叙事站点，阅读时应区分 **个人贡献边界**。

## 关联页面

- [GR00T-VisualSim2Real](./gr00t-visual-sim2real.md)
- [人形机器人](./humanoid-robot.md)
- [Sim2Real](../concepts/sim2real.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [Teleoperation](../tasks/teleoperation.md)
- [SONIC（规模化运动跟踪）](../methods/sonic-motion-tracking.md)
- [Zhengyi Luo（罗正宜）](./zhengyi-luo.md)
- [Xue Bin Peng（彭学斌）](./xue-bin-peng.md)

## 参考来源

- [Tairan He 个人主页原始资料](../../sources/sites/tairan-he.md)

## 推荐继续阅读

- [VIRAL 项目页](https://viral-humanoid.github.io/)
- [OmniH2O 项目页](https://omni.human2humanoid.com/)
