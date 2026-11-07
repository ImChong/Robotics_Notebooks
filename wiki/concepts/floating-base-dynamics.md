---
type: concept
status: complete
original:
  - ../../sources/papers/herzog_2016_momentum_control_hierarchical_inverse_dynamics_arxiv_1410_7284.md
  - ../../sources/papers/modern_robotics_textbook.md
sources:
  - ../../sources/papers/herzog_2016_momentum_control_hierarchical_inverse_dynamics_arxiv_1410_7284.md
  - ../../sources/papers/modern_robotics_textbook.md
  - ../../sources/papers/robot_kinematics_tools.md
  - ../../sources/papers/diebel_2006_representing_attitude_quaternions.md
  - ../../sources/papers/robocompiler_humanoids_composable_dynamics_arxiv_2610_14237.md
summary: "Floating Base Dynamics 描述底座不固定机器人在自由空间和接触条件下的整体动力学组织方式。"
updated: 2026-11-07
---

# Floating Base Dynamics

**Floating Base Dynamics（浮动基动力学）**：描述机器人在基座不固定于世界坐标系时，其整体动力学如何建模与控制的框架。

## 一句话定义

如果说固定底座机械臂的基座是“焊死在地上”的，那人形、四足、无人机这类机器人就不是这样。

它们的躯干 / base 会随系统整体运动，因此：

> **Floating Base Dynamics** 研究的就是：当机器人“根部不固定”时，整机动力学该怎么建模、怎么理解、怎么控制。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TSID | Task-Space Inverse Dynamics | 任务空间逆动力学求解关节力矩的 WBC 实现 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| LIP | Linear Inverted Pendulum | 线性倒立摆，质心动力学的常用简化模型 |
| ZMP | Zero Moment Point | 足式平衡判据，地面反力合力矩为零的点 |
| IMU | Inertial Measurement Unit | 惯性测量单元，提供加速度与角速度 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |

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

## 核心原理：动力学方程长什么样

先看固定基座的对照（*Modern Robotics* Ch 8，开链）：$\tau=M(\theta)\ddot\theta+h(\theta,\dot\theta)$，末端对环境施力时右端加 $J^\top(\theta)F_{tip}$；$M(\theta)=\sum_iJ_{ib}^\top\mathcal G_iJ_{ib}$ 对称正定，$\mathcal G_i$ 为连杆 $i$ 的 $6\times6$ 空间惯量。**教材的动力学只含固定基座**，下面的浮动基方程是它加上未驱动的基座自由度和接触力后的推广。

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

（Herzog et al. 2016 式 1 写作 $M(q)\ddot q+N(q,\dot q)=S^\top\tau+J_c^\top\lambda$，$N$ 含科氏 / 离心 / 重力 / 摩擦，$\lambda$ 为广义接触力；其 $S=[I_{n\times n}\ 0]$ 把 $n$ 个关节放在前、6 个浮动基自由度放在后，本页上文把基座写在前，只是排列不同。）

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

> **原文核对说明：** 本节 RNEA / CRBA / ABA 与空间向量代数部分出自 Featherstone《Rigid Body Dynamics Algorithms》及 Pinocchio 等工具的通行说法，**本批原文核对未取得该书**，其复杂度与符号细节标「（未核对原书）」。已核对的相关内容是 *Modern Robotics* Ch 8 的固定基座牛顿–欧拉递推：前向迭代 $T_{i,i-1}=e^{-[A_i]\theta_i}M_{i,i-1}$，$\mathcal V_i=\mathrm{Ad}_{T_{i,i-1}}(\mathcal V_{i-1})+A_i\dot\theta_i$，$\dot{\mathcal V}_i=\mathrm{Ad}_{T_{i,i-1}}(\dot{\mathcal V}_{i-1})+\mathrm{ad}_{\mathcal V_i}(A_i)\dot\theta_i+A_i\ddot\theta_i$；后向迭代 $\mathcal F_i=\mathrm{Ad}^\top_{T_{i+1,i}}(\mathcal F_{i+1})+\mathcal G_i\dot{\mathcal V}_i-\mathrm{ad}^\top_{\mathcal V_i}(\mathcal G_i\mathcal V_i)$，$\tau_i=\mathcal F_i^\top A_i$；重力作为基座加速度 $\dot{\mathcal V}_0=(0,-g)$ 引入。浮动基推广把基座当作带 6 自由度的根关节。

Featherstone（2008）给出了浮动基系统的三个递推算法（未核对原书的复杂度说法：RNEA、ABA 为 $O(n)$，CRBA 为 $O(n^2)$ 量级），是 Pinocchio/RBDL 等工具的核心：

### RNEA（Recursive Newton-Euler Algorithm）— 逆动力学

**输入**：$q, \dot{q}, \ddot{q}$（广义位置/速度/加速度）  
**输出**：关节力矩 $\tau$（教材固定基座版见上；浮动基版输出含根关节的 6 维力）

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

复杂度 $O(n^2)$ 量级（未核对原书），输出惯性矩阵 $M \in \mathbb{R}^{(n+6)\times(n+6)}$。固定基座下教材给出等价的 $M(\theta)=\sum_iJ_{ib}^\top\mathcal G_iJ_{ib}$。

### ABA（Articulated Body Algorithm）— 正动力学

**输入**：$q, \dot{q}, \tau$  
**输出**：$\ddot{q}$（关节加速度）

复杂度 $O(n)$（未核对原书），比直接求逆 $M$ 快得多，常用于仿真器时间积分；教材的固定基座前向动力学则是直接解 $M(\theta)\ddot\theta=\tau-h-J^\top F_{tip}$ 的线性方程。

---

## 空间向量代数（Spatial Vector Algebra）

Featherstone 的空间向量代数将 3D 力学量统一为 6D 向量，简化推导：

| 空间量 | 维度 | 组成 |
|--------|------|------|
| 空间速度（twist）| 6 | $[ω; v]$（角速度 + 线速度） |
| 空间加速度 | 6 | $[\dot{ω}; \dot{v}]$ |
| 空间力（wrench）| 6 | $[τ; f]$（力矩 + 力） |
| 空间惯性 | 6×6 | 质心系下 $\mathcal G=\mathrm{diag}(\mathcal I_b,\,mI)$（教材）；换系 $\mathcal G_a=[\mathrm{Ad}_{T_{ba}}]^\top\mathcal G_b[\mathrm{Ad}_{T_{ba}}]$ |

> 修正：本页旧版把空间惯性写成 $\begin{bmatrix}I_{cm}+mcc^\top & mc\\ mc^\top & mE\end{bmatrix}$，该形式缺少叉乘矩阵（应为 $c$ 的反对称矩阵 $[c]$ 而非向量 $c$），已改为教材的质心系形式。教材排序为 twist $(\omega,v)$、wrench $(m,f)$（角 / 力矩在前）。

空间运动方程（教材单刚体 twist–wrench 形式）：

$$\mathcal F_b=\mathcal G_b\dot{\mathcal V}_b-[\mathrm{ad}_{\mathcal V_b}]^\top\mathcal G_b\mathcal V_b$$

教材强调：只要 $\mathcal F,\mathcal V,\mathcal G$ 都在同一坐标系里表达，方程形式不随坐标系而变。Featherstone 的写法 $I^A\dot V+v\times^*I^AV=F$ 与此同构（符号不同，未核对原书）；「Pinocchio 完全基于此实现」未核对其源码，按推测处理。

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

**浮动基行 = 整机牛顿–欧拉方程。** 把方程按驱动行与浮动基行拆开（Herzog et al. 2016 式 12–14）：浮动基的 6 行 $M_l\ddot q+N_l=J_{c,l}^\top\lambda$ 不含 $\tau$，正是整个系统的动量变化 = 外力（接触力 + 重力）；关节力矩只出现在驱动行，$\tau=M_u\ddot q+N_u-J_{c,u}^\top\lambda$ 由 $\ddot q,\lambda$ 唯一决定，所以可以从优化变量里消去（变量数从 $2n+6+6c$ 降到 $n+6+6c$，该论文因此让层级逆动力学跑进 1 kHz）。物理上说就是：你只能通过接触力 $\lambda$ 和关节运动间接改变基座，无法直接命令基座加速度。

**姿态参数化陷阱：** 若基座姿态用四元数，则 $n_q=n_v+1$，$\dot q$ 里的四元数导数不是角速度 $\omega$，二者由四元数率与角速度的线性关系联系（Diebel 2006 §6.7）；教材用 $\dot RR^{-1}=[\omega_s]$、$R^{-1}\dot R=[\omega_b]$ 区分空间 / 体角速度。动力学方程里的 $\dot q$ 实际是广义速度 $v$。

---

## 和 centroidal dynamics 的关系

Herzog et al. 2016 给出了二者的精确连接：质心动量 $h=H_G(q)\dot q$（$H_G$ 为 centroidal momentum matrix，$h=[h_{lin};h_{ang}]$ 取在质心），$\dot h=H_G\ddot q+\dot H_G\dot q$，同时由浮动基行得 $\dot h=\begin{bmatrix}I&0&\cdots\\ [x_i-x_{cog}]_\times&I&\cdots\end{bmatrix}\lambda+\begin{bmatrix}mg\\0\end{bmatrix}$——即 centroidal dynamics 恰是浮动基方程的这 6 行。

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

## 工程实践：在机器人中的典型应用

- **控制栈接口：** 读取 $q,\dot q$（含基座估计，见 [State Estimation](./state-estimation.md)）→ 取 $M,N,J_c$ → 层级 QP / [TSID](./tsid.md) 求 $\ddot q,\lambda$ → 由驱动行算 $\tau$（Herzog 等 1 kHz 的做法）。
- **调试检查：** 零力矩 + 零接触力时 $\ddot q=-M^{-1}N$ 必须满足动力学（自由落体）；浮动基 6 行的合力应等于 $m\ddot x_{cog}-mg$。
- **库使用：** 注意 $n_q\neq n_v$、四元数顺序与基座在前 / 在后的排布差异，各库约定不同（以所用库文档为准）。

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

## 局限与常见误区

- **本页方程假设刚体、固定接触（$J_c\ddot q+\dot J_c\dot q=0$）：** 接触建立 / 断开、冲击、滑移不在方程内（见 [Contact Dynamics](./contact-dynamics.md)）；刚体假设忽略柔性与关节摩擦细节。
- **模型误差：** 惯量参数不准会直接污染 $M,N$；Herzog 等指出模型误差与计算复杂度是实际部署的限制。

### 常见误区

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

## 关联页面

- [Centroidal Dynamics](./centroidal-dynamics.md) — 浮动基方程的 6 个未驱动行
- [Contact Dynamics](./contact-dynamics.md) — 接触力与接触约束如何进入方程
- [TSID](./tsid.md)、[Whole-Body Control](./whole-body-control.md) — 在该方程上做任务空间控制
- [State Estimation](./state-estimation.md) — 基座位姿 / 速度估计
- [Modern Robotics 教材](../entities/modern-robotics-book.md) — 固定基座动力学（Ch 8）
- [SO(3) 旋转表示对比](../comparisons/so3-rotation-representations.md) — 基座姿态表示

## 推荐使用建议

### 如果你做人形控制
这页必须懂。

### 如果你做 RL locomotion
哪怕你不直接写动力学方程，也建议理解 floating base 和 fixed base 的根本差异，不然很多 reward / observation / action 设计会很模糊。

### 如果你做 WBC / TSID / MPC
这页其实就是你的前置数学语境。

## 参考来源

- [Herzog et al. 2016，arXiv:1410.7284 原文归档](../../sources/papers/herzog_2016_momentum_control_hierarchical_inverse_dynamics_arxiv_1410_7284.md) — 浮动基运动方程、驱动 / 浮动基行拆分、质心动量矩阵
- [sources/papers/robot_kinematics_tools.md](../../sources/papers/robot_kinematics_tools.md) — ingest 档案（Pinocchio 2019 / RBDL / Crocoddyl，均基于浮动基 RNEA/CRBA/ABA）
- Featherstone, *Rigid Body Dynamics Algorithms* — 浮基动力学核心算法参考（RNEA/CRBA/ABA/空间向量代数；本批未取得原书，相关陈述未核对）
- Orin et al., *Centroidal Dynamics of a Humanoid Robot* (2013) — 人形浮基系统质心动力学（未取得原文，未核对；质心动量矩阵的定义以 Herzog 2016 引用的 Orin & Goswami 2008 为线索）
- Siciliano et al., *Robotics: Modelling, Planning and Control* — 机器人动力学建模教材（未核对）
- Lynch & Park (2017), *Modern Robotics*. Ch 8 *Dynamics of Open Chains* — 用李群 / 螺旋理论给出 Lagrange 与 Newton-Euler 形式的统一推导，本科级入口
- [sources/papers/modern_robotics_textbook.md](../../sources/papers/modern_robotics_textbook.md)
- [sources/papers/diebel_2006_representing_attitude_quaternions.md](../../sources/papers/diebel_2006_representing_attitude_quaternions.md) — 四元数率 ↔ 角速度、$n_q \neq n_v$
- [sources/papers/robocompiler_humanoids_composable_dynamics_arxiv_2610_14237.md](../../sources/papers/robocompiler_humanoids_composable_dynamics_arxiv_2610_14237.md) — 合作人形可组合约简动力学（RoboCompiler 扩展）

## 推荐继续阅读

- [Herzog et al., *Momentum Control with Hierarchical Inverse Dynamics on a Torque-Controlled Humanoid* (arXiv:1410.7284)](https://arxiv.org/abs/1410.7284) — 浮动基运动方程拆分与动量控制
- Featherstone, *Rigid Body Dynamics Algorithms*
- [Modern Robotics 教材](../entities/modern-robotics-book.md)
- [单位四元数与 SO(3)](../formalizations/unit-quaternion-so3.md) — 浮动基四元数在 $q$ 中的布局与 $\dot q$ 误区
- [Centroidal Dynamics](./centroidal-dynamics.md)
- [SE(3) 切空间浮动基 TO](../entities/paper-se3-tangent-to.md) — 浮动基 RPY / 四元数 / \(\mathfrak{se}(3)\) 在全身 TO 里的对照
- [Articulated Body Algorithms](../formalizations/articulated-body-algorithms.md) — ABA / RNEA 与 $M(q)$、$g(q)$ 计算
- [URDF Robot Description](./urdf-robot-description.md)
- [State Estimation](./state-estimation.md)
- [RoboCompiler-Humanoids](../entities/paper-robocompiler-humanoids-composable-dynamics.md) — 多人形手牵手 / 支撑切换的可组合浮动基约简动力学（显式 Coriolis + 功率恒等式；arXiv:2610.14237）

## 一句话记忆

> Floating Base Dynamics 研究的是“机器人基座不固定时，整机动力学怎么建模和控制”，它是人形、四足、WBC、centroidal dynamics 和状态估计的共同前置。
