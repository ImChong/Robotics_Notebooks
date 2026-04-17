---
type: method
---

# Trajectory Optimization

**Trajectory Optimization（轨迹优化）**：把机器人“从哪里出发、怎么运动、最终到哪里去”写成一个带动力学和约束的优化问题，求一条满足目标且代价尽量小的轨迹。

## 一句话定义

如果说控制在回答“这一刻该怎么做”，那 **Trajectory Optimization** 更像是在回答：

> 接下来一段时间，整条运动轨迹应该怎么安排，才又稳、又合理、又满足约束。

## 为什么重要

机器人很多问题都不是单步决策，而是整段过程设计。

比如：
- 机械臂从 A 到 B 怎么绕开障碍
- 双足机器人下一串落脚点怎么安排
- 起身、下蹲、跨步、跑跳的整体动作怎么生成
- 质心和接触力轨迹如何规划

这类问题如果只看当前一步，往往会很蠢。

Trajectory Optimization 重要就在于：
- 它能自然处理长时域目标
- 它能显式纳入动力学和约束
- 它是 OCP 在机器人运动生成里的核心落地形式
- 它是 MPC、motion planning、legged locomotion、whole-body motion generation 的重要基础

一句话，**它解决的是“整段动作怎么设计”而不是“这一拍怎么补救”。**

## 核心问题长什么样

轨迹优化通常会同时优化：
- 状态轨迹 \(x_0, x_1, ..., x_N\)
- 控制轨迹 \(u_0, u_1, ..., u_{N-1}\)
- 有时还包括接触时序、落脚点、相位变量等

标准形式通常写成：

$$
\min_{x_{0:N}, u_{0:N-1}} \sum_{k=0}^{N-1} l(x_k, u_k) + l_f(x_N)
$$

满足：

$$
x_{k+1} = f(x_k, u_k)
$$

以及各种约束：
- 状态约束
- 控制约束
- 碰撞约束
- 接触约束
- 终端约束

这和 OCP 本质一致，但机器人里更强调“轨迹”这件事本身。

## 它到底在优化什么

### 1. State Trajectory
状态怎么随时间演化。

比如：
- 关节角轨迹
- 质心轨迹
- base pose 轨迹
- 末端轨迹

### 2. Control Trajectory
控制输入怎么随时间变化。

比如：
- 力矩序列
- 加速度序列
- 接触力序列

### 3. Event / Contact Variables
在人形和足式机器人里，经常还要优化：
- 落脚点位置
- 接触时序
- 步长 / 步频
- 相位切换时刻

这也是它比普通轨迹插值高级很多的地方。

## 常见求解路线

### 1. Direct Shooting
只把控制序列当变量：
- 给一组 \(u\)
- 用动力学前向 rollout 出整条轨迹
- 然后优化 \(u\)

优点：
- 结构简单
- 变量少

缺点：
- 数值容易不稳定
- 对初值敏感
- 长时域问题容易炸

### 2. Multiple Shooting
把中间状态也当变量，同时加动态一致性约束。

优点：
- 比单纯 shooting 更稳
- 更适合长时域与复杂约束

缺点：
- 变量更多

### 3. Direct Collocation
把状态和控制在离散节点上一起优化，并通过 collocation 约束逼近连续动力学。

这是机器人里非常经典的一条路线。

优点：
- 数值稳定性好
- 容易加复杂约束
- 很适合 whole-body motion / legged planning

缺点：
- 问题规模大
- 实时性通常不如简化方法

## 常见算法 / 家族

### 1. iLQR / DDP
适合有连续动力学、想高效求局部最优解的场景。

特点：
- 利用二次近似
- 对轨迹优化很常见
- 在控制和运动生成里都很有影响力

### 2. SQP（Sequential Quadratic Programming）
把非线性优化问题迭代近似成一系列 QP。

适合：
- 非线性约束多
- 精度要求高

### 3. NLP Solvers
把轨迹优化问题直接交给非线性规划求解器。

常见工具：
- IPOPT
- SNOPT
- KNITRO
- CasADi（建模层）

### 4. Convex Relaxation / Convexification
对部分问题可做凸化，换取更好的实时性与鲁棒性。

在人形 / 四足 locomotion 里很常见，尤其是 centroidal-level planning。

## 在机器人中的典型应用

### 1. 机械臂运动生成
规划一条平滑、无碰撞、满足动力学或速度约束的末端运动轨迹。

### 2. 双足 / 四足步态规划
优化：
- CoM 轨迹
- 接触力
- 落脚点
- 接触时序

这在人形 locomotion 里非常核心。

### 3. 起身、跳跃、翻越等动态动作
这些动作高度非线性、约束多，通常需要 trajectory optimization 才能写清楚。

### 4. 作为 MPC 的离线先验
离线先用 trajectory optimization 找一条高质量参考轨迹，在线再由 MPC / TSID / WBC 跟踪。

## 和 MPC 的关系

这两个最容易混。

### Trajectory Optimization
更偏：
- 离线 / 中低频
- 整段动作设计
- 求一条完整参考轨迹

### MPC
更偏：
- 在线 / 高频
- 每时刻滚动优化
- 根据当前状态不断修正

关系可以理解成：

```text
Trajectory Optimization：先把大致整段路设计好
MPC：走的时候随时修正
```

很多系统里两者是组合关系，不是二选一。

## 和已有页面的关系

### 和 Optimal Control 的关系
Trajectory Optimization 本质上是最优控制在机器人运动生成问题上的直接落地。

见：[Optimal Control (OCP)](../concepts/optimal-control.md)

### 和 LQR / iLQR 的关系
iLQR 是轨迹优化的核心算法之一，Crocoddyl 中的 FDDP solver 本质上是 iLQR 的扩展。

见：[LQR / iLQR](../formalizations/lqr.md)

### 和 MPC 的关系
MPC 可以看作 trajectory optimization 的在线滚动版本之一。

见：[Model Predictive Control (MPC)](./model-predictive-control.md)

### 和 Centroidal Dynamics 的关系
在足式 / 人形机器人里，trajectory optimization 常常在 centroidal level 上规划 CoM、momentum、contact force 和 footstep。

见：[Centroidal Dynamics](../concepts/centroidal-dynamics.md)

### 和 TSID / WBC 的关系
Trajectory Optimization 给出参考轨迹，TSID / WBC 负责把这些参考真正执行到关节层。

见：[TSID](../concepts/tsid.md)

见：[Whole-Body Control](../concepts/whole-body-control.md)

## 常见坑

### 1. 初值依赖严重
很多非凸问题很容易卡在局部最优，没有好初值就很痛苦。

### 2. 模型不准
优化出来的轨迹在仿真里很美，真机一上就垮，根源常常是模型偏差。

### 3. 约束太多导致求解困难
尤其接触切换、碰撞、摩擦锥这些加进去后，问题会迅速变重。

### 4. 只追求数学最优，不顾可执行性
轨迹优化给出的解不一定容易跟踪，所以常常要兼顾：
- 动态可行性
- 控制可跟踪性
- 对扰动的鲁棒性

## 继续深挖入口

如果你想沿着 trajectory optimization 继续往下挖，建议从这里进入：

### 论文入口
- [Survey Papers](../../references/papers/survey-papers.md)
- [Whole-Body Control 论文导航](../../references/papers/whole-body-control.md)

### 工具 / Repo 入口
- [Utilities](../../references/repos/utilities.md)
- [Humanoid Projects](../../references/repos/humanoid-projects.md)

## 参考来源

- Tedrake, *Underactuated Robotics* — trajectory optimization 入门（MIT 课程，Chapter 10）
- Kelly, *An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation* (2017) — 直接配置法综述
- Posa et al., *Optimization and stabilization of trajectories for constrained dynamical systems* (2014) — 接触轨迹优化
- **ingest 档案：** [sources/papers/robot_kinematics_tools.md](../../sources/papers/robot_kinematics_tools.md) — Crocoddyl（多接触 TO 求解器）

## 推荐继续阅读

- Tedrake, *Underactuated Robotics*（trajectory optimization 章节）
- Kelly, *An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation*
- Posa et al., *Optimization and stabilization of trajectories for constrained dynamical systems*

## 关联页面

- [Optimal Control (OCP)](../concepts/optimal-control.md) — 轨迹优化是 OCP 的数值实现方式
- [Model Predictive Control (MPC)](./model-predictive-control.md) — MPC 是滚动执行的有限域轨迹优化
- [LQR / iLQR](../formalizations/lqr.md) — iLQR 是轨迹优化的特殊高效实现
- [Whole-Body Control](../concepts/whole-body-control.md) — 轨迹优化常作为 WBC 的上层规划器
- [Drake](../entities/drake.md) — Drake 是轨迹优化的主流工具之一

## 一句话记忆

> Trajectory Optimization 解决的是”整段动作轨迹怎么设计”，它把机器人运动生成写成一个带动力学和约束的优化问题，是连接最优控制、MPC、步态规划和全身动作生成的核心桥梁。
