---
type: concept
---

# Floating Base Dynamics

**Floating Base Dynamics（浮动基动力学）**：描述机器人在基座不固定于世界坐标系时，其整体动力学如何建模与控制的框架。

## 一句话定义

如果说固定底座机械臂的基座是“焊死在地上”的，那人形、四足、无人机这类机器人就不是这样。

它们的躯干 / base 会随系统整体运动，因此：

> **Floating Base Dynamics** 研究的就是：当机器人“根部不固定”时，整机动力学该怎么建模、怎么理解、怎么控制。

## 为什么重要

这页看起来像基础概念，但其实是人形控制里最容易被跳过、又最不能跳过的一页。

原因很简单：

- 人形机器人不是机械臂
- 机械臂的 base 固定在地上
- 人形机器人的躯干位置和姿态会随接触与动作一起变化
- 所以你不能直接把它当普通关节链来做控制

一句话：

> 只要你研究人形、四足、双足 locomotion、全身控制、centroidal dynamics、TSID / WBC，这一层都绕不过去。

## 什么叫 floating base

### 固定底座（fixed base）
比如桌面机械臂：
- 机器人底盘固定在世界坐标系
- base 的位置和姿态已知且不变
- 只需要考虑关节自由度

典型状态：

$$
q = [q_1, q_2, ..., q_n]
$$

### 浮动底座（floating base）
比如：
- 人形机器人
- 四足机器人
- 双足机器人
- 无人机机械臂系统

这类系统中：
- 躯干 / 根节点本身可以平移和旋转
- base 不是直接被 actuator 驱动的关节
- 它通过接触力、关节运动和整体动力学间接变化

所以状态通常写成：

$$
q = [q_b, q_j]
$$

其中：
- \( q_b \)：floating base 的位姿（通常 6 自由度）
- \( q_j \)：关节自由度

## 为什么它和固定底座完全不是一回事

### 1. base 不是直接可控量
在人形机器人里，你不能像控制一个关节那样直接“给 base 一个电机命令”。

base 的运动是通过：
- 关节动作
- 接触力
- 地面反力
- 整体动力学耦合

共同产生的。

### 2. 整体动力学会强耦合
固定底座系统里，很多事情更局部。

floating base 系统里：
- 摆腿会影响躯干
- 接触力会改变整体动量
- 手臂摆动也会反过来影响平衡

这就是为什么人形控制天然更难。

### 3. 接触决定可控性
机器人 base 能不能稳住，不只是看关节控制器，而是看：
- 当前有没有接触
- 接触在哪里
- 接触力够不够
- 摩擦是否允许

所以 floating base 系统的控制几乎天然和 contact dynamics 绑在一起。

## 典型状态表示

在浮动基机器人里，常见广义坐标表示为：

$$
q = [x_b, y_b, z_b, \phi_b, \theta_b, \psi_b, q_1, ..., q_n]
$$

也就是：
- 基座位置（3）
- 基座姿态（3）
- 关节角（n）

广义速度常见写法：

$$
\dot{q} = [v_b, \omega_b, \dot{q}_j]
$$

其中：
- \( v_b \)：base 线速度
- \( \omega_b \)：base 角速度
- \( \dot{q}_j \)：关节速度

## 动力学方程长什么样

浮动基机器人的经典动力学通常写成：

$$
M(q) \ddot{q} + h(q, \dot{q}) = S^T \tau + J_c^T f
$$

这里：
- \( M(q) \)：质量矩阵
- \( h(q, \dot{q}) \)：重力、科里奥利、离心项
- \( \tau \)：关节力矩
- \( S \)：选择矩阵（只对 actuated joints 有控制输入）
- \( J_c \)：接触雅可比
- \( f \)：接触力

这个式子和固定底座最不一样的点在于：

> **不是所有自由度都直接有 actuator。**

floating base 的 6 个自由度通常是 **unactuated**。

这也是为什么要引入选择矩阵 \( S \)。

## 这意味着什么

### 1. 你不能直接命令 base 加速度
你不能说：
- “base 往前加速一点”
- 然后就像关节那样直接实现

你必须通过：
- 关节运动
- 接触力分配
- 整体动量调整

来间接产生这个效果。

### 2. 接触力是控制链的一部分
对机械臂来说，很多时候末端接触只是任务场景。

对人形来说：
- 接触力本身就是系统动力学的核心变量
- 地面反力不是“环境反馈”，而是 base motion 的关键来源

### 3. 平衡控制本质上是浮动基控制
很多“平衡问题”其实就是：
- 如何在 unactuated base 的前提下
- 通过可控关节和接触力
- 维持整体姿态、质心与动量稳定

## 核心算法：RNEA / CRBA / ABA

Featherstone（2008）给出了浮动基系统的三个 O(n) 递推算法，是 Pinocchio/RBDL 等工具的核心：

### RNEA（Recursive Newton-Euler Algorithm）— 逆动力学

**输入**：$q, \dot{q}, \ddot{q}$（广义位置/速度/加速度）  
**输出**：关节力矩 $\tau$

递推流程（两遍扫描）：
```
前向扫描（根→叶）：
  计算每个链节的空间速度和加速度
  计算惯性力和科氏力

后向扫描（叶→根）：
  从叶节点向上累积力
  输出每个关节处的力矩
```

浮动基时，base 的 6D 力矩通常不直接对应 actuator，需配合选择矩阵 $S$。

### CRBA（Composite Rigid Body Algorithm）— 惯性矩阵

**计算** $M(q)$，用于后续动力学方程求解：

$$M(q)\ddot{q} + h(q,\dot{q}) = S^T\tau + J_c^T f$$

复杂度 $O(n^2)$，输出惯性矩阵 $M \in \mathbb{R}^{(n+6)\times(n+6)}$。

### ABA（Articulated Body Algorithm）— 正动力学

**输入**：$q, \dot{q}, \tau$  
**输出**：$\ddot{q}$（关节加速度）

复杂度 $O(n)$，比直接求逆 $M$ 快得多，常用于仿真器时间积分。

---

## 空间向量代数（Spatial Vector Algebra）

Featherstone 的空间向量代数将 3D 力学量统一为 6D 向量，简化推导：

| 空间量 | 维度 | 组成 |
|--------|------|------|
| 空间速度（twist）| 6 | $[ω; v]$（角速度 + 线速度） |
| 空间加速度 | 6 | $[\dot{ω}; \dot{v}]$ |
| 空间力（wrench）| 6 | $[τ; f]$（力矩 + 力） |
| 空间惯性 | 6×6 | $I^A = \begin{bmatrix} I_{cm} + mcc^T & mc \\ mc^T & mE \end{bmatrix}$ |

空间运动方程简化为：

$$I^A \dot{V} + v \times^* I^A V = F$$

其中 $v \times^*$ 是空间力的伴随算子。这种表示消除了大量符号混乱，Pinocchio 完全基于此实现。

---

## 浮动基的约束结构

浮动基的 EOM（运动方程）在有接触时展开为：

$$\begin{bmatrix} M_{ff} & M_{fj} \\ M_{jf} & M_{jj} \end{bmatrix} \begin{bmatrix} \ddot{q}_f \\ \ddot{q}_j \end{bmatrix} + \begin{bmatrix} h_f \\ h_j \end{bmatrix} = \begin{bmatrix} 0 \\ \tau \end{bmatrix} + J_c^T f$$

- 左上 $M_{ff}$：base 自身惯性（6×6）
- 右下 $M_{jj}$：关节惯性（n×n）
- 耦合块 $M_{fj}$：base-关节交叉耦合
- $h$ 包含重力、科氏力、离心力
- 约束 $J_c \ddot{q} = -\dot{J}_c \dot{q}$（接触加速度为零）

WBC/TSID 的核心任务就是在满足此 EOM 和接触约束的前提下分配 $\tau$ 和 $f$。

---

## 和 centroidal dynamics 的关系

Floating Base Dynamics 是更通用、更底层的整机动力学视角。

Centroidal Dynamics 可以看作：
- 从 full-body floating base dynamics 里抽取出
- 对质心和整体动量最关键的一层中间模型

所以关系大概是：

```text
Floating Base Full-Body Dynamics
    ↓（抽取与平衡最相关的量）
Centroidal Dynamics
    ↓（进一步简化）
LIP / ZMP
```

见：[Centroidal Dynamics](./centroidal-dynamics.md)

见：[LIP / ZMP](./lip-zmp.md)

## 和 TSID / WBC 的关系

TSID / WBC 其实很多时候就是在处理 floating base 系统的控制问题。

它们在做的事情是：
- 接受任务空间目标
- 满足 floating base 动力学一致性
- 同时满足接触约束
- 输出关节加速度 / 力矩 / 接触力

所以你可以把这页理解成：

> TSID / WBC 的动力学前置。

见：[TSID](./tsid.md)

见：[Whole-Body Control](./whole-body-control.md)

## 和 State Estimation 的关系

floating base 系统还有一个麻烦点：

- base pose 和 base velocity 常常不能直接测
- 需要靠 IMU、编码器、接触信息去估计

所以：
- fixed-base 机械臂的状态估计往往比较简单
- floating-base 人形机器人里，状态估计是核心难点之一

见：[State Estimation](./state-estimation.md)

## 在机器人中的典型应用

### 1. 人形机器人 locomotion
只要你研究：
- 单脚支撑
- 双脚支撑
- 摆腿
- 扰动恢复

本质上都在处理 floating base dynamics。

### 2. 四足机器人
四足虽然不是人形，但同样是 floating base 系统。

### 3. 全身操作
如果机器人一边站着、一边用手操作，那 floating base dynamics 和接触约束都会一起上来。

### 4. 高动态动作
跳跃、跑动、翻越等动作里，floating base dynamics 的影响会更强。

## 常见误区

### 1. 以为只是在状态里多了 6 个自由度
不止。真正难的是：
- 这 6 个自由度不是直接可控的
- 它们和关节、接触、整体动量耦合得非常强

### 2. 以为人形控制只是“大一点的机械臂控制”
完全不是。floating base + contact 才是本质区别。

### 3. 以为只要会 WBC / TSID 就不用懂 floating base dynamics
不行。你可以不天天手推公式，但得知道这些方法到底在解决什么动力学结构问题。

### 4. 低估 contact 的重要性
在 floating base 系统里，contact 不是附加项，而是系统动力学的一部分。

## 推荐使用建议

### 如果你做人形控制
这页必须懂。

### 如果你做 RL locomotion
哪怕你不直接写动力学方程，也建议理解 floating base 和 fixed base 的根本差异，不然很多 reward / observation / action 设计会很模糊。

### 如果你做 WBC / TSID / MPC
这页其实就是你的前置数学语境。

## 参考来源

- [sources/papers/robot_kinematics_tools.md](../../sources/papers/robot_kinematics_tools.md) — ingest 档案（Pinocchio 2019 / RBDL / Crocoddyl，均基于浮动基 RNEA/CRBA/ABA）
- Featherstone, *Rigid Body Dynamics Algorithms* — 浮基动力学核心算法参考（RNEA/CRBA/ABA/空间向量代数）
- Orin et al., *Centroidal Dynamics of a Humanoid Robot* (2013) — 人形浮基系统质心动力学
- Siciliano et al., *Robotics: Modelling, Planning and Control* — 机器人动力学建模教材

## 推荐继续阅读

- Featherstone, *Rigid Body Dynamics Algorithms*
- [Centroidal Dynamics](./centroidal-dynamics.md)
- [TSID](./tsid.md)
- [State Estimation](./state-estimation.md)

## 一句话记忆

> Floating Base Dynamics 研究的是“机器人基座不固定时，整机动力学怎么建模和控制”，它是人形、四足、WBC、centroidal dynamics 和状态估计的共同前置。
