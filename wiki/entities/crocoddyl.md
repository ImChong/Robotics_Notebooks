---
type: entity
sources:
  - ../../sources/papers/optimal_control.md
summary: "Crocoddyl"
updated: 2026-05-16
---

# Crocoddyl

**Crocoddyl** 是一个面向机器人最优控制与轨迹优化的开源工具箱，长期由 **LAAS-CNRS / INRIA / Gepetto / Stack-of-Tasks** 这条学术与开源路线推动。

## 一句话定义

如果说 Pinocchio 提供的是机器人运动学、动力学和导数的高质量计算底座，那 **Crocoddyl** 提供的就是：

> 一套建立在 Pinocchio 之上的最优控制与 trajectory optimization 工具链，尤其适合做多自由度机器人和人形机器人的 shooting-based optimal control。

一句话说白了：

> `Crocoddyl` 是把“动力学模型”真正变成“可求解最优控制问题”的那层工具箱。

## 为什么重要

做 trajectory optimization / optimal control 的时候，论文里看起来通常很抽象：
- 定义状态和控制
- 写代价函数
- 写动力学约束
- 跑 DDP / FDDP / shooting method

但真正落到工程里，你需要的是：
- 能表达复杂机器人动力学
- 能快速做 rollout
- 能高效算导数
- 能组织 state / action / cost / residual / actuation / contact model
- 能稳定求解 shooting-based optimal control

Crocoddyl 重要的地方就在于：

- 它把这些最优控制工程细节组织成了库
- 它特别适合 legged / humanoid / manipulation 场景
- 它和 Pinocchio 配合得非常紧密
- 它几乎是 model-based robot control 工具栈里最值得学的一套开源方案之一

## 它到底是什么

### 1. 不是仿真器
Crocoddyl 不负责像 MuJoCo / Isaac Gym 那样搭世界、推进物理仿真时间步。

它不是环境平台，而是：
- trajectory optimization toolbox
- optimal control problem builder
- shooting solver 工具库

### 2. 不是现成控制器
它也不是像 TSID / WBC 那样“拿来就输出关节力矩”的低层控制器。

它更偏：
- 离线规划
- 中低频优化
- 参考轨迹生成
- optimal control research

所以它通常站在控制链更前面。

## 它在解决什么问题

### 1. 给定模型，求一条最优轨迹
最典型问题：
- 机器人从 A 动到 B
- 同时满足动力学和约束
- 代价尽量小（能量、误差、时间等）

这就是 trajectory optimization / optimal control 的经典问题。

### 2. 组织复杂机器人最优控制问题
对于人形 / 足式机器人，问题会变得很复杂：
- 浮动基
- 多接触
- 动力学非线性
- 多目标代价
- 接触切换

Crocoddyl 提供了一套较成熟的建模和求解结构，来组织这些问题。

### 3. 用 shooting-based 方法高效求解
Crocoddyl 最核心的气质就是：
- shooting methods
- DDP / FDDP / Gauss-Newton 风格
- 适合机器人动力学优化

这让它和很多一般性 NLP 求解框架有不同的使用味道。

## 为什么它在机器人 optimal control 里很强

### 1. 和 Pinocchio 深度绑定
这点特别关键。

Crocoddyl 之所以强，不是因为它凭空实现了一切，而是因为：
- Pinocchio 负责高质量动力学与导数计算
- Crocoddyl 在这个基础上做 optimal control 建模与求解

这让它非常适合高自由度机器人。

### 2. 对 legged / humanoid 场景友好
人形 / 足式最怕的就是：
- 浮动基
- 接触
- 非线性动力学
- 大维度状态

Crocoddyl 在这些场景里长期有很强代表性，特别适合：
- walking motion optimization
- jumping / crouching / recovery motion
- loco-manipulation planning

### 3. 很适合研究型工作流
如果你在做：
- research prototype
- algorithm verification
- model-based control baseline
- trajectory optimization paper reproduction

Crocoddyl 非常顺手。

## 它的典型能力

### 1. State / Actuation / Differential Action Model 组织
它把最优控制问题拆成很多结构化模块：
- 状态模型
- 驱动模型
- 动力学模型
- 代价模型
- residual
- terminal model

这对组织复杂机器人问题特别有帮助。

### 2. DDP / FDDP 求解器
Crocoddyl 很重要的一条线是：
- DDP（Differential Dynamic Programming）
- FDDP（Feasibility-driven DDP）

这类方法在机器人 optimal control 里非常常见。

### 3. 接触和冲击建模
做人形 / 足式轨迹优化时：
- 单脚支撑
- 双脚支撑
- 接触切换
- 冲击建模

都很关键。Crocoddyl 在这方面有很强的针对性。

### 4. 与 Pinocchio 配套的导数效率
如果做 trajectory optimization，导数质量直接决定：
- 收敛速度
- 稳定性
- 工程可维护性

Crocoddyl 在这点上吃到 Pinocchio 很多红利。

## 它和当前项目主线的关系

### 和 Trajectory Optimization 的关系
这几乎是最直接的关系。

Crocoddyl 是 trajectory optimization 在机器人场景里的代表性工具箱之一。

见：[Trajectory Optimization](../methods/trajectory-optimization.md)

### 和 Optimal Control 的关系
Crocoddyl 是把最优控制问题变成工程可解问题的实践工具。

见：[Optimal Control (OCP)](../concepts/optimal-control.md)

### 和 Pinocchio 的关系
Pinocchio 给它提供运动学、动力学和导数底座，Crocoddyl 在其之上做 shooting-based optimal control。

见：[Pinocchio](./pinocchio.md)

### 和 MPC 的关系
Crocoddyl 通常不等于 MPC，但它的求解思路、模型组织方式和很多 nonlinear MPC 有很强亲缘关系。

见：[Model Predictive Control (MPC)](../methods/model-predictive-control.md)

### 和 Centroidal Dynamics / WBC 的关系
Crocoddyl 可以用于更高层的轨迹优化与运动规划，再由 WBC / TSID 执行；也可以直接在 full-body / contact 场景中做更复杂的动作优化。

见：[Centroidal Dynamics](../concepts/centroidal-dynamics.md)

见：[Whole-Body Control](../concepts/whole-body-control.md)

## 它和 TSID / WBC 的区别

这三个很容易混。

### Crocoddyl
更偏：
- trajectory optimization
- optimal control
- 规划整段运动
- 中低频 / 离线 / 参考轨迹生成

### TSID / WBC
更偏：
- 低层任务执行
- 约束一致控制
- 高频闭环
- 把参考落成关节加速度 / 力矩 / 接触力

一句话：

> Crocoddyl 更像“先把整条动作想明白”，TSID / WBC 更像“现在这一拍怎么稳稳执行出来”。

## 常见误区

### 1. 以为 Crocoddyl 是控制器
不是，它更像最优控制求解与建模工具箱。

### 2. 以为学会 Crocoddyl 就等于学会 trajectory optimization
不够。它是非常好的工具，但方法论和问题建模仍然是核心。

### 3. 以为它只能做人形
不对。机械臂、足式、操作任务也能用。

### 4. 以为它和 Pinocchio 是替代关系
完全不是。更准确地说：
- Pinocchio 是底座
- Crocoddyl 是上层 optimal control 工具箱

## 推荐使用建议

### 如果你做轨迹优化 / optimal control
非常值得学。

尤其是：
- humanoid motion planning
- legged locomotion optimization
- model-based baseline
- floating-base nonlinear control

### 如果你做 WBC / TSID
也值得理解 Crocoddyl，因为很多时候它负责给你更高层的参考轨迹。

### 如果你主要做 RL
不一定非得把它用熟，但理解它能帮助你更清楚：
- model-based 方法能做到什么
- RL 和 optimal control 的边界在哪里

## 推荐继续阅读

- 官方仓库：<https://github.com/loco-3d/crocoddyl>
- 文档：<https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/master/doxygen-html/>
- 论文：Mastalli et al., *Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control*
- [Pinocchio](./pinocchio.md)

## 参考来源

- Mastalli et al., *Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control* (2020) — Crocoddyl 论文
- 官方仓库：<https://github.com/loco-3d/crocoddyl>

## 关联页面

- [Pinocchio](./pinocchio.md)
- [cuRobo](./curobo.md) — GPU 并行碰撞与多样本 TO 的另一条实现谱系（与 shooting/DDP 工具链问题剖分不同）
- [Optimal Control](../methods/model-predictive-control.md)
- [Whole-Body Control](../concepts/whole-body-control.md)

## 一句话记忆

> Crocoddyl 是建立在 Pinocchio 之上的机器人最优控制与轨迹优化工具箱，特别适合 legged / humanoid 场景，是把动力学模型真正变成可求解 optimal control 问题的关键一层。
