# Pinocchio

**Pinocchio** 是机器人领域最主流的刚体运动学与动力学计算库之一，长期由 **Stack-of-Tasks / INRIA / LAAS-CNRS** 这条学术与开源路线推动。

## 一句话定义

如果说 MuJoCo、Isaac Gym 这类工具是在做“仿真”，那 **Pinocchio** 做的是：

> 高效、稳定地计算机器人运动学、动力学、雅可比、质心量、逆动力学以及它们的导数。

一句话说白了：

> `Pinocchio` 是机器人控制和优化工具链里最重要的底层数学引擎之一。

## 为什么重要

很多控制和优化方法，论文里只写：
- 计算正逆动力学
- 求雅可比
- 算质心动量矩阵
- 算导数

但真落到代码里，这些东西需要一个：
- 快
- 准
- 稳
- 支持复杂机器人
- 还能做导数计算

的底层库。

Pinocchio 之所以重要，就是因为它长期在做这件事，而且做得非常强。

它的重要性不在于“它本身是一个完整框架”，而在于：

> 它经常是完整框架下面最关键的一层。

## 它到底是什么

### 1. 不是仿真器
Pinocchio 不负责把机器人放进物理世界里跑时间步仿真。

它不是：
- MuJoCo
- Isaac Gym
- Isaac Lab
- Gazebo

它更像是“刚体模型计算内核”。

### 2. 不是控制器
Pinocchio 也不是 MPC / WBC / TSID 本身。

它不直接告诉你“下一步该施加什么控制”，但它会给你这些控制器最需要的底层量：
- forward kinematics
- inverse kinematics 支撑量
- Jacobian
- mass matrix
- inverse dynamics
- centroidal quantities
- derivatives

所以它不是最上层方法，但它常常是最关键的底座。

## Pinocchio 在解决什么问题

### 1. 运动学计算
你给它：
- 机器人模型
- 当前关节状态

它帮你算：
- 各 link 姿态
- 末端位姿
- Jacobian
- frame transform

### 2. 动力学计算
它还能算：
- mass matrix
- inverse dynamics（RNEA）
- forward dynamics（ABA）
- non-linear effects
- centroidal dynamics 相关量

### 3. 导数与优化支撑
这块是 Pinocchio 很强的一点：
- 对很多动力学量支持高效导数计算
- 很适合 trajectory optimization / optimal control / DDP / MPC
- 是很多 model-based 工具链偏爱它的核心原因

## 为什么它在机器人控制里很重要

### 1. 它是“算动力学”的标准底座之一
在机器人领域，很多方法最终都需要：
- 速度和加速度怎么传播
- 一个关节力矩对系统的影响
- 末端运动和关节运动的映射关系
- 浮动基系统的动力学量

Pinocchio 把这些都做成了高质量库。

### 2. 它很适合人形 / 足式机器人
人形机器人会带来：
- 浮动基
- 高自由度
- 多接触
- 质心与动量相关量计算

Pinocchio 对这些问题特别有价值，因为它在 humanoid / WBC / TSID / Crocoddyl 生态里长期深度使用。

### 3. 它是很多上层框架的共同底座
如果你学人形控制，很容易一路遇到：
- TSID
- Crocoddyl
- whole-body optimization
- model-based RL / optimal control 工具链

这些系统很多都直接或间接依赖 Pinocchio。

所以理解 Pinocchio，不只是学一个库，而是在理解：

> 这些高级控制和优化框架下面，真正拿来算动力学的东西是什么。

## 它的典型能力

### 1. Forward Kinematics
给定关节角，算 link / frame 的位置姿态。

### 2. Jacobian
求：
- 关节速度 → 末端速度
- 广义速度 → frame motion

这对：
- inverse kinematics
- operational space control
- TSID / WBC

都很关键。

### 3. Inverse Dynamics
经典 RNEA：
- 给定状态、速度、加速度
- 求需要的关节力矩

### 4. Forward Dynamics
经典 ABA：
- 给定状态、速度、关节力矩
- 求系统加速度

### 5. Centroidal Quantities
它很适合算：
- CoM
- centroidal momentum
- centroidal momentum matrix

这对人形控制特别关键。

### 6. Derivatives
这是 Pinocchio 在优化控制里非常吃香的一点。

如果你做：
- DDP / iLQR
- trajectory optimization
- nonlinear MPC
- Crocoddyl

你会非常在意导数是不是：
- 准
- 快
- 可维护

Pinocchio 在这点上非常强。

## 它和当前项目主线的关系

### 和 Centroidal Dynamics 的关系
当前项目主线上有很重要的一环是 centroidal dynamics。

Pinocchio 可以高效提供很多质心与动量相关计算，是这条线往代码里落的重要底座。

见：[Centroidal Dynamics](../concepts/centroidal-dynamics.md)

### 和 TSID 的关系
TSID 这种任务空间逆动力学框架，需要大量动力学、雅可比和导数计算。

Pinocchio 正是这类框架最典型的底层依赖之一。

见：[TSID](../concepts/tsid.md)

### 和 Whole-Body Control 的关系
WBC 框架虽然站在更高层，但它背后常常离不开：
- Jacobian
- inverse dynamics
- centroidal quantities
- contact frame 相关计算

这些都是 Pinocchio 擅长的。

见：[Whole-Body Control](../concepts/whole-body-control.md)

### 和 Trajectory Optimization / MPC 的关系
做 trajectory optimization 和 model-based control 时，Pinocchio 的导数支持非常重要。

见：[Trajectory Optimization](../methods/trajectory-optimization.md)

见：[Model Predictive Control (MPC)](../methods/model-predictive-control.md)

## 它和 MuJoCo / Isaac Gym 的区别

### MuJoCo / Isaac Gym / Isaac Lab
更偏：
- 仿真平台
- 环境执行
- RL 训练底座

### Pinocchio
更偏：
- 模型计算
- 运动学 / 动力学内核
- 控制与优化底座

所以别把它们放在一个维度上硬比。

更准确地说：
- MuJoCo / Isaac Gym 是“让机器人在世界里跑”
- Pinocchio 是“帮你算机器人本体的数学量”

## 常见误区

### 1. 以为 Pinocchio 是仿真器
不是，它不负责完整物理仿真世界。

### 2. 以为 Pinocchio 是控制器
也不是，它是控制器和优化器依赖的底层计算库。

### 3. 以为会用 Pinocchio 就等于会做 WBC / MPC
不够。Pinocchio 只是底座，方法论和问题建模还得自己掌握。

### 4. 低估导数支持的重要性
如果你只做简单 kinematics，可能感觉不到；但一旦进入 optimal control / trajectory optimization / DDP，这点的重要性会瞬间上来。

## 推荐使用建议

### 如果你做控制 / 优化
Pinocchio 很值得优先学。

尤其是：
- 人形控制
- 足式机器人
- centroidal dynamics
- trajectory optimization
- TSID / WBC

### 如果你做 RL
哪怕你主要做 RL，也建议理解 Pinocchio 在 model-based baseline 里的作用。这样你对“RL 和控制的边界”会更清楚。

### 如果你是初学者
最好的入门方式不是把 API 背下来，而是：
- 先用它跑一个简单机械臂 forward kinematics
- 再算 Jacobian
- 再算 inverse dynamics
- 再慢慢接到 humanoid / floating-base 场景

## 推荐继续阅读

- 官方仓库：<https://github.com/stack-of-tasks/pinocchio>
- 官方文档：<https://stack-of-tasks.github.io/pinocchio/>
- [TSID](../concepts/tsid.md)
- [Crocoddyl](./crocoddyl.md)

## 参考来源

- [sources/papers/robot_kinematics_tools.md](../../sources/papers/robot_kinematics_tools.md) — ingest 档案（Pinocchio 2019 / Crocoddyl 2020 / RBDL 2017）
- Carpentier et al., *Pinocchio: Fast Forward and Inverse Dynamics for Poly-articulated Systems* (2019) — Pinocchio 论文
- 官方仓库：<https://github.com/stack-of-tasks/pinocchio>
- 官方文档：<https://stack-of-tasks.github.io/pinocchio/>

## 关联页面

- [Crocoddyl](./crocoddyl.md)
- [TSID](../concepts/tsid.md)
- [Whole-Body Control](../concepts/whole-body-control.md)

## 一句话记忆

> Pinocchio 不是仿真器，也不是控制器，它是机器人运动学、动力学和导数计算的底层引擎，是 TSID、WBC、trajectory optimization 和 humanoid control 工具链里的关键基础设施。
