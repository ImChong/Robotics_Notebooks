---
type: method
tags: [dexterity, manipulation, robot-hand, reinforcement-learning, control]
status: complete
updated: 2026-07-11
related:
  - ../entities/allegro-hand.md
  - ../entities/shadow-hand.md
  - ../tasks/manipulation.md
  - ../concepts/tactile-sensing.md
  - ../formalizations/cross-modal-attention.md
  - ./uhas-unified-hand-action-space.md
sources:
  - ../../sources/papers/imitation_learning.md
  - ../../sources/papers/uhas_arxiv_2607_03570.md
summary: "手内重定向（In-hand Reorientation）是指机器人灵巧手在不借助于外部环境（如桌面）的前提下，通过手指间的协同动作改变掌心中物体位姿的技术。"
---

# In-hand Reorientation (手内重定向)

**手内重定向 (In-hand Reorientation)** 是灵巧操作（Dexterous Manipulation）领域中最具挑战性的任务之一。它的目标是让多指灵巧手（如 Allegro Hand 或 Shadow Hand）在不松开物体、且不利用桌面等外部支撑的情况下，仅依靠手指间的动态协作，将物体从初始姿态调整到预期的目标姿态。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 核心挑战

1. **频繁的接触切换 (Hybrid Dynamics)**：为了让物体旋转，手指必须交替进行“支撑-滑动-抬起-再接触”的循环，系统动力学具有极强的离散非线性。
2. **感知遮挡 (Perception Occlusion)**：在手内旋转时，手指会频繁遮挡物体，导致基于视觉的位姿估计失效。
3. **欠驱动与滑移**：由于接触点面积有限，物体极易发生意外滑脱（Slip），需要极高频的力反馈调节。

## 主要方法路线

### 1. 强化学习 (Deep RL)
目前最成功的手内重定向成果多来自于无模型强化学习（Model-free RL）。
- **代表作**：OpenAI 的 *Learning Dexterous In-Hand Manipulation*。
- **机制**：在仿真中通过海量试错发现奇特的“指尖舞蹈”策略。结合**域随机化 (Domain Randomization)** 解决 Sim2Real 问题。
- **跨具身动作空间**：[UHAS](./uhas-unified-hand-action-space.md) 把手内立方体重定向策略建在 **规范球面形变** 上，用 **单一 PPO 策略** 同时服务 Allegro、LEAP、Shadow、MANO 四手，并支持零样本迁移与快速微调（arXiv:2607.03570）。

### 2. 轨迹优化 (Trajectory Optimization)
将重定向建模为带接触约束的最优控制问题。
- **代表作**：基于 Drake 的接触隐式优化（Contact-Implicit TO）。
- **机制**：显式地计算接触力和手指关节路径，物理意义明确但计算开销巨大。

### 3. 基于触觉的闭环控制
利用触觉传感器提供的接触点漂移信息进行实时补偿。
- **关联**：[触觉感知 (Tactile Sensing)](../concepts/tactile-sensing.md)。

## 典型应用场景

- **拧螺丝**：在受限空间内，通过手指转动螺丝。
- **魔方还原**：展示极致的协调性。
- **工具使用**：调整手中工具（如锤子或笔）的抓握方向以适配后续任务。

## 关联页面
- [Allegro Hand 实体](../entities/allegro-hand.md)
- [Shadow Hand 实体](../entities/shadow-hand.md)
- [UHAS（统一手部动作空间）](./uhas-unified-hand-action-space.md)
- [Manipulation 任务](../tasks/manipulation.md)
- [Tactile Sensing (触觉感知)](../concepts/tactile-sensing.md)
- [Cross-modal Attention](../formalizations/cross-modal-attention.md)

## 参考来源
- OpenAI, et al. (2018). *Learning Dexterous In-Hand Manipulation*.
- Akkaya, I., et al. (2019). *Solving Rubik’s Cube with a Robot Hand*.
