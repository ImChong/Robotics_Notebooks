---
type: method
tags: [control, lqr, ilqr, optimal-control, optimization, locomotion]
status: complete
updated: 2026-04-25
related:
  - ../formalizations/lqr.md
  - ./trajectory-optimization.md
  - ./model-predictive-control.md
  - ../entities/crocoddyl.md
  - ../concepts/optimal-control.md
summary: "LQR / iLQR 是机器人最优控制与轨迹优化的基石，通过 Riccati 递归高效求解线性及非线性轨迹优化问题。"
---

# LQR / iLQR 算法详解

**LQR (Linear Quadratic Regulator)** 是线性最优控制的解析基石，而 **iLQR (iterative LQR)** 是其在非线性系统上的威力延伸。它们通过贝尔曼最优性原理（Bellman Optimality）和 Riccati 递归，将复杂的轨迹优化问题转化为高效的线性代数迭代。

## 一句话定义

> LQR 是线性系统下的“一步到位”解析解；iLQR 是非线性系统下的“反复横跳”迭代优化，是现代高性能轨迹优化器（如 Crocoddyl）的核心引擎。

---

## 1. LQR (Linear Quadratic Regulator)

### 核心逻辑：逆向 Riccati 递归
对于离散线性系统 $x_{k+1} = Ax_k + Bu_k$ 和二次代价 $J = \sum (x^T Q x + u^T R u)$，LQR 通过从终端 $T$ 向初始时刻 $0$ 逆向推导 **Riccati 方程** 来获取最优反馈增益 $K_k$。

**算法步骤：**
1. **初始化**：令 $P_T = Q_f$（终端代价矩阵）。
2. **逆向递推** ($k = T-1 \dots 0$)：
   - 计算增益：$K_k = (R + B^T P_{k+1} B)^{-1} B^T P_{k+1} A$
   - 更新代价矩阵：$P_k = Q + A^T P_{k+1} (A - B K_k)$
3. **在线执行**：
   - 最优控制输入：$u_k^* = -K_k x_k$

---

## 2. iLQR (Iterative LQR)

iLQR 专门处理非线性动力学 $x_{k+1} = f(x_k, u_k)$。它通过在当前参考轨迹附近进行**二阶展开（代价函数）**和**一阶展开（动力学）**，反复调用 LQR 逻辑来逼近非线性最优解。

### 算法闭环：Backward & Forward Pass

#### Step 1: Backward Pass (寻找改进方向)
在参考轨迹上计算动力学雅可比 ($f_x, f_u$) 和代价函数的导数 ($l_x, l_u, l_{xx}, l_{uu}, l_{ux}$)。
从 $k=T-1$ 到 $0$ 计算：
- **Q-值近似**：
  - $Q_x = l_x + f_x^T V'_x$
  - $Q_u = l_u + f_u^T V'_x$
  - $Q_{xx} = l_{xx} + f_x^T V'_{xx} f_x$
  - $Q_{uu} = l_{uu} + f_u^T V'_{xx} f_u$
  - $Q_{ux} = l_{ux} + f_u^T V'_{xx} f_x$
- **计算更新增益**：
  - 反馈项：$K_k = -Q_{uu}^{-1} Q_{ux}$
  - 前馈项：$k_k = -Q_{uu}^{-1} Q_u$
- **更新价值函数梯度** ($V_x, V_{xx}$) 传给下一时刻。

#### Step 2: Forward Pass (应用改进并滚出新轨迹)
使用更新步长 $\alpha \in (0, 1]$（通过线搜索 Line Search 确定）：
$$u_k^{new} = u_k^{old} + \alpha k_k + K_k (x_k^{new} - x_k^{old})$$
$$x_{k+1}^{new} = f(x_k^{new}, u_k^{new})$$

---

## 3. iLQR vs DDP (Differential Dynamic Programming)

iLQR 常被称为“DDP 的简化版”。

| 特性 | iLQR | DDP |
|------|------|-----|
| **动力学展开** | 一阶（Jacobians） | 二阶（Hessians $f_{xx}, f_{uu}, f_{ux}$） |
| **计算开销** | 较低，不需要求动力学二阶导 | 较高，需要二阶导 |
| **收敛速度** | 接近二阶收敛 | 严格二阶收敛 |
| **复杂性** | 适合大多数机器人任务 | 适合极其精细或高度非线性的任务 |

---

## 主要分类

- **理论底座**：[最优控制 (OCP)](../concepts/optimal-control.md) / [Bellman 方程](../formalizations/bellman-equation.md) / [HJB 方程](../formalizations/hjb.md)
- **数值核心**：Riccati 递归 (LQR) / 迭代线性化 (iLQR)
- **工程实现**：[Crocoddyl](../entities/crocoddyl.md) (iLQR/FDDP) / [Drake](../entities/drake.md) (Direct Collocation/LQR)
- **上层形态**：[MPC 模型预测控制](./model-predictive-control.md) / [轨迹优化](./trajectory-optimization.md)

---

## 4. 实践中的 Trick (Engineering Know-How)

### 正则化 (Regularization)
在计算 $Q_{uu}^{-1}$ 时，如果 $Q_{uu}$ 接近奇异（比如处于非物理状态或过大控制步长），会导致数值崩溃。
- **做法**：给 $Q_{uu}$ 加上一个小常数 $\mu I$。
- **意义**：类似 Levenberg-Marquardt 算法，在牛顿步和梯度下降步之间切换。

### 线搜索 (Line Search)
由于 iLQR 基于局部线性化，如果步子迈得太大，非线性动力学可能会发散。
- **做法**：从 $\alpha=1$ 开始尝试，如果新轨迹的代价没有下降（Armijo Condition），则减小 $\alpha$（如 $\alpha \leftarrow 0.5 \alpha$）。

### 处理约束 (Constraints)
标准 iLQR 不支持不等式约束。
- **做法 A (Box-iLQR)**：在 Backward Pass 求解 $u$ 时考虑控制限位 $\underline{u} \le u \le \bar{u}$（通常用简单的钳位或 QP 求解）。
- **做法 B (Penalty/Barrier)**：将约束以惩罚项形式加入代价函数。

---

## 5. 在机器人中的应用场景

1. **足式运动生成**：计算从站立到行走的质心和足端轨迹。
2. **灵巧操作规划**：在 MuJoCo 中规划手指拨动物体的复杂序列。
3. **全身控制器内核**：作为 MPC 的底层求解器，每 10-20ms 运行一次以应对扰动。

## 参考来源

- Li & Todorov, *Iterative Linear Quadratic Regulator Design for Control of Nonlinear Biological Movement Systems* (2004) — iLQR 经典论文。
- Tassa et al., *Control-Limited Differential Dynamic Programming* (2014) — 解决带约束的 iLQR。
- [Optimal Control Course (CMU 16-745)](../../references/README.md) — 理论推导背景。
- **ingest 档案：** [sources/papers/optimal_control.md](../../sources/papers/optimal_control.md)

## 关联页面

- [LQR / iLQR 形式化](../formalizations/lqr.md) — 理论背景与数学推导
- [Trajectory Optimization](./trajectory-optimization.md) — 轨迹优化全景
- [Crocoddyl](../entities/crocoddyl.md) — 基于 iLQR/FDDP 的工业级开源求解器
- [MPC](./model-predictive-control.md) — LQR 在线滚动的形态
