---
type: formalization
tags: [locomotion, control, dynamics, humanoid, math]
status: complete
updated: 2026-11-07
related:
  - ../concepts/lip-zmp.md
  - ../concepts/centroidal-dynamics.md
  - ../methods/model-predictive-control.md
  - ../tasks/locomotion.md
  - ./friction-cone.md
original:
  - ../../sources/papers/lip_zmp_dcm_walking_reading_primary_papers.md
sources:
  - ../../sources/papers/lip_zmp_dcm_walking_reading_primary_papers.md
  - ../../sources/papers/mpc.md
summary: "ZMP + LIP（Linear Inverted Pendulum）模型通过简化质心高度和角动量，将人形机器人平衡问题转化为可解析的线性动力学方程。"
---

# ZMP + LIP 形式化

## 一句话定义

**LIP (Linear Inverted Pendulum)** 模型是将人形机器人简化为在固定高度平面内运动的质点，而 **ZMP (Zero Moment Point)** 则是判断该系统是否平衡的关键指标。本页给出二者的公式化推导；概念性介绍见 [LIP / ZMP](../concepts/lip-zmp.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ZMP | Zero Moment Point | 足式平衡判据，地面反力合力矩为零的点 |
| LIP | Linear Inverted Pendulum | 线性倒立摆，质心动力学的常用简化模型 |
| CoM | Center of Mass | 质心，平衡与 locomotion 规划的核心状态量 |
| DCM | Divergent Component of Motion | 质心发散分量，用于落脚点与平衡调节 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| QP | Quadratic Programming | 将 WBC/控制问题写成二次规划的标准求解形式 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 为什么重要

质心高度恒定 + 忽略角动量后，CoM 的水平动力学变成线性、解耦、可解析求解的方程，ZMP 又是质心加速度的线性函数；于是“下一步 ZMP 在哪里、CoM 会怎样”可以写成线性状态空间，直接套 LQR、预览控制和线性 MPC（QP），每个控制周期亚毫秒级求解（Herdt 等 2010 报告完整 QP 平均 < 1 ms）。

## 核心原理：核心假设

1. **质心高度恒定**：质心（CoM）在 $z$ 方向上的位置固定为 $z_c$。
2. **忽略角动量**：忽略由于躯干摆动产生的角动量变化 $\dot{L}_G \approx 0$。
3. **共面支撑**：接触位于同一水平面，接触力满足单边接触与摩擦约束；点足的支撑区域退化为点，平足及双支撑可以形成有限支撑多边形。

## LIP 动力学方程

在这些假设下，质心在水平面（$x, y$ 方向）上的运动方程变得完全解耦且线性：

$$
\ddot{x} = \omega^2 (x - x_{zmp})
$$
$$
\ddot{y} = \omega^2 (y - y_{zmp})
$$

其中：
- $\omega = \sqrt{g / z_c}$ 是倒立摆的固有频率。
- $g$ 是重力加速度。
- $(x, y)$ 是质心位置。
- $(x_{zmp}, y_{zmp})$ 是零力矩点（ZMP）位置。

**反向关系与离散预测模型（Herdt 等 2010 式 (1)–(5)）：** 方程等价于 $x_{zmp} = x - \ddot{x}/\omega^2 = x - (z_c/g)\ddot{x}$，即 ZMP（此处取作 CoP 的近似）由质心位置与加速度线性给出。取状态 $\hat{x}=[x,\dot{x},\ddot{x}]^\top$、输入为分段常值 jerk $\dddot{x}$（周期 $T$），则

$$
\hat{x}_{k+1} = A\hat{x}_k + B\dddot{x}_k,\quad
A=\begin{bmatrix}1&T&T^2/2\\0&1&T\\0&0&1\end{bmatrix},\;
B=\begin{bmatrix}T^3/6\\T^2/2\\T\end{bmatrix},\quad
z_k=\begin{bmatrix}1&0&-z_c/g\end{bmatrix}\hat{x}_k
$$

递推 $N$ 步得 $Z_{k+1}=P_{zs}\hat{x}_k+P_{zu}\dddot{X}_k$，代价取 $\frac{\alpha}{2}\|\dddot{X}\|^2+\frac{\gamma}{2}\|Z-Z^{ref}\|^2$ 即得到 Kajita 式预览控制所对应的 QP（Herdt 式 (11)–(16)）。

**DCM 分解（Romualdi 等 2018 式 (2)–(3)）：** 令 $\xi = x+\dot{x}/\omega$，则 $\dot{x}=\omega(\xi-x)$，$\dot{\xi}=\omega(\xi-x_{zmp})$：CoM 稳定地追随 DCM，DCM 自身有正实特征根 $\omega$。

## ZMP 稳定性条件

ZMP 是支撑平面上接触合力矩的水平分量为零的点；在平面接触且合力法向分量非零时，它与整体压力中心对应。在共面、单边接触的简化模型下，接触可行性要求：

$$
\mathbf{p}_{zmp} \in \mathcal{S}_{support\_polygon}
$$

即 ZMP 位于支撑多边形（Support Polygon）内。该瞬时条件不能单独保证长期稳定或可停止；还需考虑质心速度、摩擦、执行器能力及未来落脚点。ZMP 接近边缘意味着接触力矩裕量减小。

## 捕获点（Capture Point, CP）

基于 LIP 方程，可以导出**捕获点（Capture Point）**。在固定高度 LIP 模型中，CP 是选择为恒定 ZMP 后可使质心渐近停止的位置：

$$
\mathbf{\xi} = \mathbf{x} + \frac{1}{\omega} \dot{\mathbf{x}}
$$

在固定支撑、固定高度且忽略角动量变化的 LIP 假设下，CP 位于支撑多边形外意味着仅调整当前支撑区域内的 ZMP 无法实现不跨步捕获；需要改变支撑或采用模型之外的平衡机制，不能直接推断真实机器人必然跌倒。

## 工程实践：在控制中的应用

1. **预览控制（Preview Control）**：通过给定的足迹序列预先规划 ZMP 轨迹，再通过求解上述线性差分方程获得最优 CoM 轨迹。
2. **DCM 控制（Divergent Component of Motion）**：将 CoM 运动分解为稳定分量和不稳定分量（即 CP），通过控制 CP 的运动来实现实时平衡。
3. **线性 MPC**：将上述方程作为预测模型，在 QP 求解器中施加 ZMP 边界约束，实现鲁棒步行。Herdt 等 2010 把落脚点也作为决策变量，在 HRP-2 仿真中只给参考速度即可自动决定步位；预测时域 $NT=1.6$ s（约两步），ZMP 约束留 3 cm 安全余量。
4. **步位 / 步时 QP（Khadiv 等 2016）：** 用 $\tau=e^{\omega T}$ 换元，把步时纳入线性等式约束，只看下一步即可得到小规模 QP；对点足 / 被动踝机器人，调步位比调 CoP 更有效。

## 局限与风险

- 假设来自模型：质心高度恒定、角动量可忽略、共面接触；高度变化、手臂摆动、跑跳、非共面支撑下偏差会变大。
- ZMP 在多边形内只是接触可行的必要条件，不保证长期稳定；Romualdi 等 2018 在 iCub 上发现带 ZMP 约束的预测控制最高直行速度 0.19–0.22 m/s，反而低于无约束的瞬时 DCM 控制（0.41 m/s），原因是预测得到的期望 ZMP 变化快，对 ZMP 跟踪误差更不鲁棒。
- 线性 MPC 的“安全优先”意味着在不可行指令或强扰动下速度跟踪被牺牲；Herdt 等记录了机器人不允许交叉腿时横向平均速度仅约指令的 2/3。
- 水平面假设下 ZMP 无法直接用于陡坡，见 [LIP / ZMP](../concepts/lip-zmp.md) 对 HumoSlope 的说明。

## 关联页面
- [Capture Point / DCM](../concepts/capture-point-dcm.md)
- [仿真物理保真度链路选型指南](../queries/simulation-physics-fidelity.md) — 本页所述物理/仿真要素在保真度链路（建模 ① → 数值 ② → 接触 ③ → 随机化 ④）中的定位
- [LIP / ZMP 概念](../concepts/lip-zmp.md)
- [Centroidal Dynamics](../concepts/centroidal-dynamics.md)
- [Model Predictive Control](../methods/model-predictive-control.md)
- [Locomotion 任务](../tasks/locomotion.md)
- [Friction Cone](./friction-cone.md)

## 参考来源
- **原文通读归档：** [LIP / ZMP / DCM 四篇一手论文通读](../../sources/papers/lip_zmp_dcm_walking_reading_primary_papers.md)（Herdt 2010、Romualdi 2018、Shafiee 2019、Khadiv 2016 全文已读；Kajita 2001 / 2003、Wieber 2006 原文未取得，下列二者为二手引述）
- Kajita, S., et al. (2001). *The 3D Linear Inverted Pendulum Mode: A simple modeling for a biped robot*.
- Wieber, P.-B. (2006). *Trajectory Free Linear Model Predictive Control for Stable Walking*.

## 推荐继续阅读

- Herdt et al., *Online Walking Motion Generation with Automatic Foot Step Placement*（开放版 <https://inria.hal.science/inria-00391408/document>）— 本页公式的完整推导与仿真
- Romualdi et al., *A Benchmarking of DCM Based Architectures for Position and Velocity Controlled Walking of Humanoid Robots* <https://arxiv.org/abs/1809.02167>
- [Model Predictive Control](../methods/model-predictive-control.md)
