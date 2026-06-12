---
type: method
title: ZEST (Zero-shot Embodied Skill Transfer)
tags: [robot-learning, humanoid, locomotion, atlas, sim2real, multi-contact]
summary: "ZEST 是波士顿动力开发的统一框架，通过自适应采样与自动课程学习，将异构动捕/视频数据直接转化为机器人的零样本高动态运动技能。"
updated: 2026-06-12
related:
  - ./mtrg-reference-goal-driven-rl.md
  - ./hil-hybrid-imitation-learning.md
  - ../concepts/curriculum-learning.md
sources:
  - ../../sources/papers/zest.md
---

# ZEST (Zero-shot Embodied Skill Transfer)

**ZEST** 是由 Boston Dynamics 团队开发的一套统一的具身技能学习与迁移框架。它通过强化学习（RL）将多样化的、异构的人类运动数据（如动捕、视频、动画）转化为机器人的高动态、多接触运动技能，并实现了在全尺寸人形机器人（Atlas）及多种机器人形态上的**零样本（Zero-shot）**真机部署。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| MDP | Markov Decision Process | 状态–动作–奖励–转移的标准序贯决策建模框架 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |

## 核心理念

ZEST 的核心在于将运动数据视为物理正则化项，在不需要显式判别器（Discriminator）或复杂接触计划的前提下，学习如何平衡“模仿精度”与“物理鲁棒性”。它证明了即使是带噪声的单目视频数据，也可以转化为机器人极其稳健的运动技能。

## 主要技术路线

```text
异构运动数据 (MoCap/ViCap/Animation)
          ↓
  物理正则化强化学习 (Adaptive Sampling)
          ↓
  自动课程学习 (Assistive Wrench)
          ↓
  极简部署策略 (No History/No Preview)
          ↓
     零样本跨形态部署 (Atlas/G1/Spot)
```

## 关键技术

### 1. 自适应采样 (Adaptive Sampling)
在处理大规模异构运动片段时，传统随机采样容易导致“灾难性遗忘”（模型在学习难动作时忘记了简单动作）。
- **EMA 失败率跟踪**：为数据集中的每个片段维护一个失败率的指数移动平均（EMA）。
- **难度感知采样**：采样器偏向于失败率高的“难点”片段，确保训练资源集中在尚未掌握的动作上。

### 2. 虚拟辅助扳手 (Virtual Assistive Wrench)
为了让机器人学会空翻等极高动态动作，ZEST 引入了一种自动课程学习机制：
- 在训练初期，在机器人基座上施加一个虚拟的外部辅助力，帮助其维持平衡。
- 该力随训练进度的提升（或通过固定计划）自动衰减至零。

### 3. 极简部署架构 (Minimalist MDP)
与大多数依赖观测历史或未来窗口的 RL 策略不同，ZEST 的策略输入被极度精简：
- **输入**：当前本体感知信号 + 下一步参考状态 + 上一步动作。
- **动作空间**：输出关节目标的残差（Residuals）并叠加在参考动作上。
这种设计最大限度地减少了仿真与现实之间的时域和观测差异。

## 性能表现

- **多接触能力**：成功实现了战术爬行、地板舞、翻滚等涉及膝盖、肘部、躯干多点接触的动作，这在传统 MPC 框架下极难建模。
- **跨形态部署**：在 Atlas、Unitree G1 和 Spot 上均实现了成功部署，且不需要针对特定硬件进行算法结构的修改。
- **高动态性**：实现了连续后空翻、侧手翻等接近人类极限的体育（Athletic）动作。

## 与 HIL / MTRG 的关系（同作者脉络）

| 工作 | 焦点 | 参考在部署时的角色 |
|------|------|-------------------|
| [HIL](./hil-hybrid-imitation-learning.md) | 物理角色跑酷动画 | tracking + AMP 并行；仿真角色 |
| **ZEST** | 多源异构模仿 → 硬件零样本 | **下一步参考**作为策略输入 |
| [MTRG](./mtrg-reference-goal-driven-rl.md) | G1 箱式跑酷 OOD 泛化 | **仅 goal**；参考只参与训练奖励 |

MTRG 复用 ZEST 的 **assistive-wrench \(\lambda\) 课程**，并在 beyond-nominal 初始条件下相对「ZEST mocap + 参考输入跟踪」显著提升成功率（见 MTRG 论文 Table I）。

## 关联页面

- [MTRG](./mtrg-reference-goal-driven-rl.md) — 参考塑形 + goal 部署的人形跑酷泛化
- [HIL](./hil-hybrid-imitation-learning.md) — tracking + AMP 混合模仿（角色动画）
- [Curriculum Learning（课程学习）](../concepts/curriculum-learning.md) — 虚拟辅助力的核心理论
- [EFGCL](./efgcl.md) — 学术侧「外部辅助力 + 课程衰减」与稀疏奖励高动态 RL（RA-L 2026 项目页 ingest）
- [Sim2Real](../concepts/sim2real.md) — 零样本迁移的实现基础
- [Behavior Cloning](./behavior-cloning.md)
- [DeepMimic](./deepmimic.md)
- [Whole-Body Control (WBC)](../concepts/whole-body-control.md)
- [Mujoco (物理引擎)](../entities/mujoco.md)
- [Atlas (机器人)](../entities/boston-dynamics.md)
- [G1 (机器人)](../entities/unitree-g1.md)
- [Spot (机器人)](../entities/boston-dynamics.md)

## 参考来源
- [机器人论文阅读笔记：ZEST](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/ZEST__Zero-shot_Embodied_Skill_Transfer_for_Athletic_Robot_Control/ZEST__Zero-shot_Embodied_Skill_Transfer_for_Athletic_Robot_Control.html)
- [ZEST: Zero-shot Embodied Skill Transfer for Athletic Robot Control](../../sources/papers/zest.md)
- [arXiv:2602.00401v1](https://arxiv.org/abs/2602.00401v1)
- [Boston Dynamics Technical Blog](https://bostondynamics.com/)
