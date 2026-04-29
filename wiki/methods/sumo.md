---
type: method
title: Sumo (MPC-over-RL 层级控制)
tags: [robot-learning, mpc, loco-manipulation, whole-body-control, spot, g1]
---

# Sumo (Dynamic and Generalizable Whole-Body Loco-Manipulation)

**Sumo** 是一种由 RAI Institute 提出的层级化机器人控制框架，专门用于解决**动态全身移动操作 (Whole-Body Loco-Manipulation)** 问题。它打破了传统的“高层学习 + 低层模型控制”范式，采用了一种**“高层 MPC 驱动底层 RL”**的反向层级架构，使机器人能够操纵比自身更重、更大或几何形状复杂的物体。

## 核心架构：MPC-over-RL

Sumo 的创新在于其分工明确的双层结构：

### 1. 底层：通用 RL WBC 策略
- **角色**：负责高频（50Hz）的全身稳定与基础移动。
- **作用**：底层是一个预训练的通用全身控制（WBC）策略（如 *Relic*）。它将高度不稳定、非线性的足式机器人动力学抽象化，转化为一个稳定的**命令空间 (Command Space)**，如基座线速度/角速度、机械臂末端目标位姿。
- **优势**：利用了强化学习在处理复杂接触和鲁棒控制方面的天然优势。

### 2. 高层：基于采样的在线 MPC
- **角色**：负责中频（20Hz）的任务规划与长时程协调。
- **方法**：使用交叉熵方法 (CEM) 进行在线轨迹采样。
- **机制**：MPC 不直接输出电机力矩，而是在底层 RL 策略的“命令空间”内进行搜索。
- **Policy-in-the-Loop**：在前向模拟（Rollouts）时，MPC 将底层 RL 策略作为动力学的一部分进行并行模拟（通常使用 MuJoCo）。

## 主要优势

- **零样本任务泛化**：由于 RL 策略是通用的，只需在部署时更换 MPC 的代价函数 (Cost Function) 或物体模型，即可让机器人执行全新的操作任务（如从推桌子切换到抬轮胎），无需任何重新训练。
- **处理超限载荷**：通过全身各部位（躯干、四肢）的协同接触，Sumo 能够操纵超过机械臂额定载荷的物体（例如让 Spot 扶起 15kg 的轮胎）。
- **极简工程代价**：相比端到端 RL 需要数十个精调的 Reward 项，Sumo 只需 3-5 个简单的几何代价项即可完成任务。

## 硬件验证

- **Spot (四足)**：在真实世界中完成了 8 项极具挑战性的任务，包括拖拽大型路障和堆叠重物。
- **G1 (人形)**：在仿真中证明了该架构在双足人形平台上的通用性，成功执行了开门和推重物等全身协调动作。

## 与其他系统的关系

- **对比 [[whole-body-control]] (WBC)**：Sumo 将 WBC 的实现交给了鲁棒的 RL 策略，而将全局约束和物体动力学交给 MPC 处理。
- **对比 [[vla]]**：Sumo 的层级架构为未来接入 VLA 模型提供了接口——VLA 可以作为更高层的推理器，动态生成 MPC 所需的 Cost Function。
- **算法依赖**：依赖于高性能并行仿真器（如 [[mujoco]]）进行实时轨迹评估。

## 参考来源
- [Sumo: Dynamic and Generalizable Whole-Body Loco-Manipulation](../../sources/papers/sumo.md)
- [RAI Institute Project Page](https://rai-institute.github.io/sumo/)
