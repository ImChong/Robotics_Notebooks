# TSID

**TSID（Task Space Inverse Dynamics，任务空间逆动力学）** 是一种典型的人形机器人全身控制方法，用来在满足动力学与接触约束的前提下，把任务空间目标转成可执行的关节加速度、力矩和接触力。

## 一句话定义

如果说 MPC / Centroidal Dynamics 在回答“机器人整体接下来该怎么动”，那 **TSID** 回答的就是：

> 这些目标具体怎么变成每个关节该出的力、该走的加速度，并且别把接触和平衡搞炸。

## 为什么重要

人形机器人控制最难的一步，不是写出一个漂亮的高层目标，而是把目标真正落到身体上。

比如你想同时做到：
- 躯干保持稳定
- 足端跟踪摆腿轨迹
- 手臂做动作
- 保持接触约束成立
- 关节力矩别超限

这些目标之间天然会冲突。

TSID 重要就在于：
- 它是 **WBC（Whole-Body Control）** 的代表性实现路线之一
- 它把任务空间目标、动力学约束、接触约束统一到一个优化框架里
- 它非常适合人形、双足、机械臂这类高自由度系统
- 它是“从规划参考到可执行全身控制”的关键落地点

一句话，**TSID 是把“我想让机器人这样动”变成“机器人真的能这样发力”的桥。**

## TSID 在解决什么问题

TSID 要解决的问题本质上是：

> 给定若干任务目标和物理约束，求一个满足动力学的一致控制解。

输出通常包括：
- 关节加速度 \( \ddot{q} \)
- 关节力矩 \( \tau \)
- 接触力 \( f \)

注意这和单纯 inverse kinematics 不一样。

- **IK** 只关心几何层怎么摆到目标位置
- **TSID** 关心动力学上能不能做到，以及力怎么出

## 核心思想

### 1. 在任务空间定义目标

常见任务包括：
- 躯干姿态跟踪
- 足端位置 / 姿态跟踪
- 手端轨迹跟踪
- CoM 跟踪
- 动量调节
- 关节姿态正则项

这些任务通常写成加速度层或力层目标，比如：

$$
\ddot{x}^* = \ddot{x}_{ref} + K_p (x_{ref} - x) + K_d (\dot{x}_{ref} - \dot{x})
$$

然后通过雅可比映射到广义坐标空间：

$$
\ddot{x} = J(q) \ddot{q} + \dot{J}(q, \dot{q}) \dot{q}
$$

也就是说，任务空间目标最后都会落到对 \( \ddot{q} \) 的约束或代价上。

### 2. 满足刚体动力学

机器人动力学方程通常写成：

$$
M(q) \ddot{q} + h(q, \dot{q}) = S^T \tau + J_c^T f
$$

其中：
- \( M(q) \)：质量矩阵
- \( h(q, \dot{q}) \)：重力、科里奥利、离心项
- \( \tau \)：关节力矩
- \( J_c \)：接触雅可比
- \( f \)：接触力
- \( S \)：驱动选择矩阵

TSID 的核心不是“猜一个动作”，而是要求这个动作必须符合真实动力学。

### 3. 满足接触约束

如果脚在地上，就不能乱滑、乱飞。

典型接触约束包括：
- 接触点加速度为零
- 法向力非负
- 摩擦锥约束
- 接触力矩限制

比如静接触约束：

$$
J_c \ddot{q} + \dot{J}_c \dot{q} = 0
$$

这保证接触点不会违反接触假设。

## 典型求解形式

TSID 常见做法是写成一个带约束的 QP（Quadratic Programming）或分层 QP。

### 单层加权 QP

把多个任务写成加权最小二乘：

$$
\min_{\ddot{q},\tau,f} \sum_i w_i \| A_i z - b_i \|^2
$$

同时满足：
- 动力学等式约束
- 接触约束
- 力矩 / 加速度 / 接触力边界

这里 \( z \) 一般是联合决策变量，例如：

$$
z = [\ddot{q}, \tau, f]
$$

### 分层 QP / Hierarchical QP

更经典的人形控制写法是优先级结构：

1. 最高优先级：动力学与接触约束
2. 次优先级：平衡 / 躯干 / 足端关键任务
3. 再次优先级：手臂动作、姿态优化、正则项

这样能处理“任务冲突”的问题。

一句话就是：

> 先保命，再走路，再优雅。

## TSID 和 WBC 的关系

TSID 不是和 WBC 并列的东西，它基本可以看作 **WBC 的代表性实现方法之一**。

关系大概是：

- **WBC**：更大的范畴，指全身协调控制思想
- **TSID**：WBC 中非常经典的一条动力学一致优化路线

所以在人形控制里你常会看到：
- WBC 框架
- TSID 求解器
- HQP controller

它们很多时候是在一个技术栈上说不同层级的东西。

## TSID 和 Centroidal Dynamics 的关系

这俩很容易混，但其实分工不同。

### Centroidal Dynamics 更像上层
它关心：
- 质心怎么动
- 整体动量怎么演化
- 接触力大概怎么分布

### TSID 更像下层执行层
它关心：
- 具体关节怎么加速
- 每个驱动器怎么出力
- 接触约束如何被精确满足

很常见的分层结构是：

```text
Centroidal Planner / MPC
    ↓
给出 CoM / momentum / footstep / contact force reference
    ↓
TSID / WBC
    ↓
输出 joint acceleration / torque / contact force
```

这就是为什么你把 `Centroidal Dynamics` 写完后，下一页就该补 `TSID`，不然控制链条会断在中间。

## TSID 和 Inverse Kinematics / Inverse Dynamics 的区别

### 和 IK 的区别
- IK：只管几何位置是否能摆到
- TSID：管动力学、接触、力矩、任务冲突

### 和传统 Inverse Dynamics 的区别
- 传统 inverse dynamics 更像“给定加速度，求力矩”
- TSID 是“在多个任务和约束下，同时求一致的加速度、力矩、接触力”

所以 TSID 不是单条公式，而是一套任务化、约束化、优化化的 inverse dynamics 框架。

## 在机器人中的典型应用

### 1. 人形机器人步行控制
最常见：
- 躯干稳住
- 支撑脚保持接触
- 摆动脚跟踪轨迹
- CoM 跟踪参考
- 关节姿态做冗余优化

### 2. 双臂 / 全身操作
当机器人一边站着一边伸手操作时，TSID 很适合同时处理：
- 手端任务
- 身体姿态稳定
- 地面接触一致性

### 3. 受约束运动控制
凡是有接触、有闭链、有多任务优先级的问题，TSID 都很强。

## 常见优点

- 动力学一致，不是只看几何
- 能自然处理多任务与优先级
- 能显式加入接触和摩擦约束
- 很适合人形 / 足式 / 高自由度系统
- 能和 MPC、centroidal planner 自然分层结合

## 常见局限

### 1. 依赖模型质量
动力学模型、接触模型、状态估计如果不准，TSID 效果会明显掉。

### 2. 实时求解压力
高频控制下解 QP/HQP 是有计算压力的，尤其任务多、约束多时。

### 3. 调参不轻松
任务权重、优先级、阻尼、正则项都需要经验，不是写完就飞。

### 4. 接触切换仍然麻烦
脚抬起、落地那一下最容易出幺蛾子，TSID 也得配合状态机、接触估计和过渡策略。

## 参考来源

- Del Prete et al., *Prioritized motion-force control of constrained fully-actuated robots: "Task Space Inverse Dynamics"* — TSID 核心论文
- Saab et al., *Dynamic whole-body motion generation under rigid contacts and other unilateral constraints* — 约束下全身动作生成
- [TSID library](https://github.com/stack-of-tasks/tsid) — 开源实现

## 和已有页面的关系

### 和 Whole-Body Control 的关系
TSID 是 WBC 最经典的实现路线之一，可以理解为“动力学一致的全身任务控制”。

见：[Whole-Body Control](./whole-body-control.md)

### 和 Centroidal Dynamics 的关系
Centroidal Dynamics 常用来做上层平衡 / 动量 / 接触规划，TSID 负责把这些参考真正落成关节层控制。

见：[Centroidal Dynamics](./centroidal-dynamics.md)

### 和 Locomotion 的关系
在双足 locomotion 里，TSID 经常负责执行摆腿、稳躯干、保接触、跟踪 CoM 参考。

见：[Locomotion](../tasks/locomotion.md)

### 和 Optimal Control / MPC 的关系
MPC 负责规划未来参考，TSID 负责在每个控制周期把这些参考变成动力学可行的低层动作。

见：[Optimal Control (OCP)](./optimal-control.md)

见：[Model Predictive Control (MPC)](../methods/model-predictive-control.md)

## 继续深挖入口

如果你想沿着 TSID 继续往下挖，建议从这里进入：

### 论文入口
- [Whole-Body Control 论文导航](../../references/papers/whole-body-control.md)

### 工具 / Repo 入口
- [Utilities](../../references/repos/utilities.md)
- [Humanoid Projects](../../references/repos/humanoid-projects.md)

## 推荐继续阅读

- Del Prete et al., *Prioritized motion-force control of constrained fully-actuated robots: “Task Space Inverse Dynamics”*
- Saab et al., *Dynamic whole-body motion generation under rigid contacts and other unilateral constraints*
- TSID library: <https://github.com/stack-of-tasks/tsid>

## 一句话记忆

> TSID 是把任务空间目标、动力学约束、接触约束和力矩分配统一起来的全身控制方法，它是人形机器人控制从“想这么动”到“真能这么动”的关键落地点。
