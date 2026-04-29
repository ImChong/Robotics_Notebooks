---
type: method
tags: [rl, imitation-learning, gan, motion-prior, humanoid]
status: complete
updated: 2026-04-27
related:
  - ../entities/mimickit.md
  - ../entities/protomotions.md
  - ./imitation-learning.md
  - ./beyondmimic.md
  - ../tasks/humanoid-soccer.md
sources:
  - ../../sources/papers/motion_control_projects.md
summary: "AMP (Adversarial Motion Prior) 通过判别器奖励引导机器人学习自然、平滑的动作风格，而 HumanX 进一步将接触图引入 AMP 框架以解决复杂的交互任务。"
---

# AMP & HumanX: 判别器驱动的风格学习

在机器人动作模仿中，单纯的轨迹跟踪奖励（如关节角度 MSE）往往会导致机器人出现高频抖动、抽搐或不自然的步态。**AMP** 引入了生成对抗的思想来提升运动质量，而 **HumanX** 将其扩展到了包含接触关系的物体交互场景。

## AMP: 对抗性动作先验

**AMP (Adversarial Motion Prior)** 的核心在于不显式定义“什么是好动作”，而是让神经网络去“悟”。

### 1. 核心架构
- **判别器 (Discriminator)**：输入一段运动片段（当前状态与历史状态），尝试区分它是来自“参考动作数据集”还是“仿真中策略生成的动作”。
- **策略 (Policy)**：作为生成器，除了最大化任务奖励外，还要最大化判别器的误判率（即让判别器认为自己生成的动作是真实的）。

### 2. 优势
- **自然度**：判别器能捕捉到人类动作中微妙的时序特征和协调性。
- **无需繁琐调参**：减少了对关节速度惩罚、平滑惩罚等启发式奖励的依赖。

## HumanX: 扩展到物体交互与接触图

**HumanX** 是对 AMP 范式的重大增强，它认为“姿态像”是不够的，“接触像”才关键。

### 1. 接触图 (Contact Graph)
HumanX 引入了接触图的概念：
- 这是一个二进制向量，标记身体各部位（左右手、左右脚、头、躯干）是否与环境或物体接触。
- **接触模仿奖励**：计算仿真接触状态与参考数据中接触状态的一致性。

### 2. 多教师蒸馏 (Multi-teacher Distillation)
HumanX 证明了学生策略可以仅仅通过**本体感知历史**（Joint angles, velocities, IMU）来隐式估计外力：
- 训练多个专注于不同技能的特权教师（使用外力、物体位姿等特权信息）。
- 学生策略通过 RL + 行为克隆 (BC) 联合训练，在没有外力传感器的情况下学会应对推力和负载。

## 主要技术路线

| 模块 | 实现方案 | 目的 |
|------|---------|------|
| **风格判别** | AMP Discriminator | 学习参考动作的自然风格，减少抖动 |
| **接触监督** | 接触图 (Contact Graph) | 确保交互任务中肢体与物体的物理一致性 |
| **技能习得** | 特权教师蒸馏 | 将外部感知能力转化为纯本体感知策略 |

## 典型奖励设计：HumanX 接触奖励

$$
r_c = \exp\left(-\sum_j \lambda_j |c_j^{sim} - c_j^{ref}|\right)
$$
其中 $c_j$ 是第 $j$ 个身体部位的接触状态（0 或 1）。该项强制机器人在特定的动作阶段（如搬箱子的抓取瞬间）保持与专家一致的物理接触。

## 参考来源

- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — 飞书公开文档《开源运动控制项目》总结。
- Peng et al., *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control*.
- [HumanX 项目主页](https://github.com/wyhuai/human-x)
- [sources/repos/amp_mjlab.md](../../sources/repos/amp_mjlab.md) — AMP 在 Unitree G1 + mjlab 上的统一 locomotion+recovery 实现。

## 关联页面
- [[protomotions]] — 提供大规模并行训练支持。

- [Imitation Learning](./imitation-learning.md)
- [Behavior Cloning](../formalizations/behavior-cloning-loss.md) — HumanX 学生策略训练中使用了 BC 损失。
- [BeyondMimic](./beyondmimic.md) — 同样是动作模仿，但 BeyondMimic 侧重于精确建模，AMP 侧重于风格判别。
- [AMP_mjlab](../entities/amp-mjlab.md) — AMP 在 Unitree G1 + mjlab 上的工程实现，统一 locomotion+recovery。

## 进阶：MimicKit 与 ADD

在 **[[mimickit]]** 框架中，AMP 得到了进一步的扩展和优化：
- **[[add]] (Adversarial Differential Discriminator)**：通过引入差分判别器，解决了 AMP 在某些场景下的滑步和运动伪影问题。
- **[[smp]] (Score-Matching Motion Priors)**：使用生成式梯度场代替传统的判别器奖励，提供了更稳定的训练信号。
