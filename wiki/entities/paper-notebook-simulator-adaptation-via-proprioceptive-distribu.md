---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2604.11090"
related:
  - ../overview/paper-notebook-category-10-sim-to-real.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_simulator-adaptation-via-proprioceptive-distribu.md
summary: "仿真训出来的腿足策略一上真机就掉点，根源是仿真与真实动力学有偏差。常见做法是去改策略（域随机化、在线适配），本文反其道：去改仿真器——让仿真更像真机，再在校准后的仿真里训策略就能直接迁移。难点在于「怎么衡量仿真像不像真机」：传统做法要逐时刻对齐轨迹，依赖动捕/特权传感、对时间对齐敏感。本文提出本体感知分布匹配（Proprioceptive Distribution Matching）：把真机与仿真各自跑一段，只看「关节观测 + 动作」的统计分布像不像，无需时间对齐、无需外部传感。用黑盒优化在这个分布距离上辨识仿真参数（或学习 action-delta / 残差执行器模型），不到 5 分钟真机数据就能显著降低漂移，效果可比肩用特权状态对齐的基线。"
---

# Simulator Adaptation for Sim-to-Real Learning of Legged Locomotion via Proprioceptive Distribution Matching

**Simulator Adaptation for Sim-to-Real Learning of Legged Locomotion via Proprioceptive Distribution Matching** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：10_Sim-to-Real），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

仿真训出来的腿足策略一上真机就掉点，根源是仿真与真实动力学有偏差。常见做法是去改策略（域随机化、在线适配），本文反其道：去改仿真器——让仿真更像真机，再在校准后的仿真里训策略就能直接迁移。难点在于「怎么衡量仿真像不像真机」：传统做法要逐时刻对齐轨迹，依赖动捕/特权传感、对时间对齐敏感。本文提出本体感知分布匹配（Proprioceptive Distribution Matching）：把真机与仿真各自跑一段，只看「关节观测 + 动作」的统计分布像不像，无需时间对齐、无需外部传感。用黑盒优化在这个分布距离上辨识仿真参数（或学习 action-delta / 残差执行器模型），不到 5 分钟真机数据就能显著降低漂移，效果可比肩用特权状态对齐的基线。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| Proprioception | 本体感知 | 机器人自身关节角/角速度/动作等内部观测，无需外部传感 |
| System ID | System Identification | 系统辨识：估计仿真器物理参数使其贴近真机 |
| Action-Delta Model | 动作增量模型 | 学一个对动作的修正量，补偿仿真-真实差异 |
| Residual Actuator Model | 残差执行器模型 | 在理想执行器之上学残差，刻画真实电机非线性 |
| Black-box Optimization | 黑盒优化 | 不需梯度、只看目标函数值的优化（如 CMA-ES 类） |

## 为什么重要

- **「改仿真」是另一条主线**：与「改策略（域随机化）」互补：把仿真校准好，下游可直接用标准 RL，流程更清晰
- **只用本体感知**：不依赖动捕/特权传感，降低系统辨识门槛，更适合现场快速校准
- **分布优于轨迹**：比较分布而非逐帧轨迹，规避时间对齐难题，对噪声/相位差更鲁棒
- **可迁人形**：方法不限四足，「少量本体数据 + 分布匹配」范式可推广到人形 sim-to-real 校准

## 解决什么问题

- **现象**：仿真训练的腿足策略在硬件上常**性能下降**，因为仿真与真实**动力学不一致**（摩擦、电机、延迟等）。 - **两条路线**：① 改**策略**（域随机化让策略变鲁棒 / 在线适配）；② 改**仿真器**（系统辨识让仿真更准）。本文走**第二条**：校准仿真器后，标准 RL 训出的策略可直接迁移。 - **传统系统辨识的痛点**：多数方法靠**逐时刻轨迹对齐**来比较仿真与真机，需要**精确时间对齐**和**特权/外部传感（动捕等）**，工程上昂贵且脆弱。

**目标**：找一种**只用机载本体感知、少量真机数据、无需时间对齐**的方式，衡量并缩小仿真-真实差距。

## 核心机制

1. **分布匹配替代轨迹匹配**：用「本体感知 (观测+动作) 的统计分布」衡量 sim-real 差距，**摆脱时间对齐与特权/外部传感**依赖，工程上更易落地。
2. **黑盒优化的仿真器自适应框架**：在分布距离上统一驱动**参数辨识 / Action-Delta / 残差执行器**三类自适应手段。
3. **极省数据**：**不到 5 分钟**真机数据即可显著降低真机漂移。
4. **可比肩特权基线**：在仅用本体感知的前提下，效果接近用「特权状态对齐」的方法；并成功适配**双腿站立行走**等高难行为。

方法拆解（深读笔记小节）：A. 核心思想：分布匹配，而非轨迹匹配；B. 三种仿真器自适应手段；C. 黑盒优化闭环。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 10_Sim-to-Real |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Simulator_Adaptation_via_Proprioceptive_Distribution_Matching/Simulator_Adaptation_via_Proprioceptive_Distribution_Matching.html> |
| arXiv | <https://arxiv.org/abs/2604.11090> |
| 机构 | Jeremy Dao、Alan Fern（Oregon State University，腿足运动学习方向） |
| 发表 | 2026-04-13（arXiv v1） |
| 项目主页 | 暂未公开 |
| 源码 | 暂未开源（截至记录日未检索到官方仓库） |
| 笔记阅读日期 | 2026-06-26 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-10-sim-to-real](../overview/paper-notebook-category-10-sim-to-real.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_simulator-adaptation-via-proprioceptive-distribu.md](../../sources/papers/humanoid_pnb_simulator-adaptation-via-proprioceptive-distribu.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Simulator_Adaptation_via_Proprioceptive_Distribution_Matching/Simulator_Adaptation_via_Proprioceptive_Distribution_Matching.html>
- 论文：<https://arxiv.org/abs/2604.11090>

## 推荐继续阅读

- [机器人论文阅读笔记：Simulator Adaptation for Sim-to-Real Learning of Legged Locomotion via Proprioceptive Distribution Matching](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Simulator_Adaptation_via_Proprioceptive_Distribution_Matching/Simulator_Adaptation_via_Proprioceptive_Distribution_Matching.html)
