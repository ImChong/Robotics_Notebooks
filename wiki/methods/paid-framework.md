---
type: method
tags: [rl, imitation-learning, perception, soccer, humanoid, unitree-g1]
status: drafting
updated: 2026-04-27
related:
  - ../tasks/humanoid-soccer.md
  - ../entities/unitree-g1.md
  - ./reinforcement-learning.md
  - ./imitation-learning.md
sources:
  - ../../sources/repos/humanoid_soccer.md
summary: "PAiD (Perception-Action integrated Decision-making) 是一种渐进式的人形机器人技能学习框架，通过模仿学习与感知-动作融合实现鲁棒的类人化踢球。"
---

# PAiD Framework

**PAiD (Perception-Action integrated Decision-making)** 是由 TeleHuman 研究团队提出的一种针对人形机器人足球技能的渐进式学习框架。其核心思想是将复杂的感知-动作任务分解为多个阶段，逐步增加系统复杂度，以解决端到端学习中常见的收敛难与泛化差的问题。

## 三阶段渐进式训练流程

PAiD 的成功关键在于其结构化的训练方案：

### 1. 动作技能习得 (Skill Acquisition)
- **目标**：获取基础的踢球肢体协调能力。
- **方法**：使用**模仿学习 (Imitation Learning)**。从人类动作捕捉数据中提取关键特征，通过 RL 训练机器人跟踪这些动作。
- **输出**：一个能够执行自然摆腿动作的基准策略（Motion Policy）。

### 2. 感知集成 (Perception Integration)
- **目标**：赋予机器人“看球”并调整动作的能力。
- **方法**：将视觉感知特征（如球的位置、速度）引入状态空间。训练策略根据感知输入动态调整第一阶段习得的动作。
- **输出**：一个能够在不同足球初始位姿下完成踢球的感知-动作策略。

### 3. 物理感知 Sim-to-Real (Physics-aware Transfer)
- **目标**：确保策略在真实物理环境（如 Unitree G1）中的有效性。
- **方法**：
    - **领域随机化**：对质量、摩擦力、视觉噪声进行大规模扰动。
    - **时延模拟**：在仿真中引入控制环路的时延，增强鲁棒性。
- **输出**：可直接部署到 Unitree G1 上的部署策略。

## 主要技术路线

| 阶段 | 核心方法 | 关键技术 |
|------|---------|---------|
| **1. 动作习得** | 模仿学习 (IL) | 基于 MoCap 数据的 [Behavior Cloning](../formalizations/behavior-cloning-loss.md) |
| **2. 感知集成** | 强化学习 (RL) | 紧耦合视觉特征的 PPO 算法 |
| **3. 真机迁移** | Sim-to-Real | [领域随机化](../concepts/domain-randomization.md) + 时延反馈补偿 |

## 技术特色

- **类人化 (Human-like)**：通过第一阶段的约束，机器人表现出更符合人类生物力学的踢球动作，而非仅仅是简单的肢体碰撞。
- **RNN 网络架构**：使用循环神经网络处理感知序列，能够更好地利用历史信息应对视觉遮挡或足球滚动。
- **高泛化性**：支持多种地形（室内地板、室外草坪）以及静止/滚动的足球。

## 应用案例

在 **Unitree G1** 机器人上的实验证明，PAiD 框架能够实现：
- 高达 90% 以上的稳健踢球成功率。
- 动态调整踢球方向和力度以射门或传球。
- 在受到轻微外部扰动时仍能维持平衡并完成动作。

## 参考来源

- [HumanoidSoccer (PAiD) 源码仓库](../../sources/repos/humanoid_soccer.md)
- *Learning Soccer Skills for Humanoid Robots: A Progressive Perception-Action Framework* (Paper)

## 关联页面

- [Humanoid Soccer](../tasks/humanoid-soccer.md)
- [Imitation Learning](./imitation-learning.md)
- [Reinforcement Learning](./reinforcement-learning.md)
- [Domain Randomization](../concepts/domain-randomization.md)
- [Behavior Cloning Loss](../formalizations/behavior-cloning-loss.md)
- [Unitree G1](../entities/unitree-g1.md)
