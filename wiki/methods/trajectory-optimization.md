---
type: method
tags: [control, optimization, motion-planning, trajectory-optimization, mpc, humanoid]
status: complete
updated: 2026-04-21
related:
  - ./model-predictive-control.md
  - ../concepts/optimal-control.md
  - ../concepts/whole-body-control.md
  - ../formalizations/zmp-lip.md
  - ../comparisons/trajectory-opt-vs-rl.md
sources:
  - ../../sources/papers/optimal_control.md
  - ../../sources/papers/mpc.md
summary: "轨迹优化（Trajectory Optimization）通过数值优化方法求解开环最优控制序列，是 MPC 的理论核心和机器狗/人形机器人离线运动规划的主力工具。"
---

# Trajectory Optimization（轨迹优化）

**轨迹优化 (Trajectory Optimization, TO)** 是一种基于动力学模型和约束条件，通过数值非线性规划（NLP）技术来自动搜索最优运动序列的计算方法。在足式机器人领域，不论是前空翻、跑酷等高难度极限动作的离线设计，还是在线模型预测控制（MPC）的底层计算核心，轨迹优化都是最不可或缺的工具。

它的核心理念是将“控制问题”转化为“数学优化问题”。只要你能用数学语言定义好机器人的物理模型和任务目标，剩下的就交给强大的数值求解器（Solver）。

## 核心形式化：最优控制问题 (OCP)

轨迹优化的本质是求解一个**最优控制问题 (Optimal Control Problem, OCP)**。给定一个初始状态 $x_0$ 和目标集合 $\mathcal{X}_f$，我们需要找到一条连续的状态轨迹 $x(t)$ 和控制指令序列 $u(t)$，以最小化某种代价函数（如能耗最小、时间最短、轨迹最平滑）：

$$ \min_{x(\cdot), u(\cdot)} \int_{0}^{T} L(x(t), u(t)) dt + \Phi(x(T)) $$
$$ \text{s.t.} \quad \dot{x}(t) = f(x(t), u(t)) \quad \text{(系统动力学约束)} $$
$$ g(x(t), u(t)) \leq 0 \quad \text{(物理限位与摩擦锥等不等式约束)} $$
$$ x(0) = x_0, \quad x(T) \in \mathcal{X}_f $$

## 三大主流求解流派

将无限维的连续 OCP 转换为有限维、可供计算机求解的 NLP 问题，通常有三种经典的离散化途径（Transcription Methods）：

### 1. 单重打靶法 (Single Shooting)
这是最直观的思路。优化变量只有**控制输入 $\mathbf{u}$**。状态轨迹 $\mathbf{x}$ 则是通过给定初始状态 $x_0$，使用当前的 $\mathbf{u}$ 在动力学模拟器中前向积分（Forward Rollout）计算得出的。
- **优点**：优化变量极少（仅包含动作），物理上绝对可行。
- **缺点**：高度非线性，极易陷入局部最优；对微小的早期控制误差极其敏感（长时域下呈指数放大，即“蝴蝶效应”）；难以处理中间过程的状态硬约束（比如中途不能碰到墙壁）。

### 2. 多重打靶法 (Multiple Shooting)
将整个时间域切分为 $N$ 个短片段。在每个短片段内单独执行打靶积分，但将每个片段的初始状态 $x_k$ 也作为独立的优化变量。此时必须增加额外的等式约束（Defect Constraints）以确保前一个片段的终点与后一个片段的起点严格缝合：$x_{k+1} = \int f(x_k, u_k) dt$。
- **优点**：大幅缓解了单重打靶的非线性发散问题，容错率高；非常适合高度并行的雅可比矩阵求导。
- **缺点**：变量数量翻倍，NLP 规模变大，依赖高级稀疏求解器（如 IPOPT, SNOPT）。

### 3. 直接配点法 (Direct Collocation)
目前最强大、在复杂多接触机器人研究中最主流的方法（例如 Drake 框架的核心机制）。它不仅把 $u_k$ 当作变量，还把所有离散点上的状态 $x_k$ 都当作**独立**的优化变量。系统动力学方程被隐式地转换为节点之间的代数等式约束（如使用三次埃尔米特插值）。
- **优点**：在处理极多状态约束（如关节限位、摩擦锥限制、防碰撞）时表现极佳；初始猜测（Initial Guess）可以即使在物理上完全不可行，求解器依然能向着可行域收敛。
- **缺点**：这是一个超大规模的 NLP 问题（数万个变量与约束），求解速度通常无法达到毫秒级的闭环要求。

## 进阶：接触隐式优化 (Contact-Implicit Trajectory Optimization)

传统轨迹优化最大的痛点是**混合系统（Hybrid System）问题**：机器人脚步接触地面的瞬间，动力学方程会发生突变。以前的做法是人为预设好哪一步左脚落地、哪一步右脚落地（Mode Sequence）。

2014 年，MIT 提出了**接触隐式优化 (Contact-Implicit Optimization)**，利用互补互斥约束（Complementarity Constraints）将接触力是否产生与足端距离地面的高度绑定在一起（距离 > 0 则力 = 0；力 > 0 则距离 = 0）。这使得优化器能够**从零开始自主发现走路的步态**，而不再需要人类手工编排时序。

## 轨迹优化的局限与未来

- **模型依赖**：由于 TO 完全是在数学模型上计算，如果真实机器人的电机摩擦、质心分布与模型有偏差（Sim2Real Gap），算出来的轨迹在实机上立刻就会跌倒。
- **在线与离线的结合**：因此，纯开环的 TO 轨迹必须配合底层的全身反馈控制（WBC），或者直接将 TO 本身以高频运行在滚动的时间窗口内（这就是 MPC）。
- **Learning to Optimize**：目前最前沿的趋势是利用深度强化学习（RL）来学习 TO 的价值函数（Value Function）或提供极佳的 Initial Guess，从而打破传统求解器的算力瓶颈。

## 关联页面
- [Model Predictive Control](./model-predictive-control.md)
- [Optimal Control (OCP) 概念](../concepts/optimal-control.md)
- [Whole-Body Control (WBC)](../concepts/whole-body-control.md)
- [ZMP + LIP 形式化](../formalizations/zmp-lip.md)
- [对比：轨迹优化 vs RL](../comparisons/trajectory-opt-vs-rl.md)

## 参考来源
- Betts, J. T. (2010). *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming*.
- Posa, M., Cantu, C., & Tedrake, R. (2014). *A direct method for trajectory optimization of rigid bodies through contact*.
- [sources/papers/optimal_control.md](../../sources/papers/optimal_control.md)
