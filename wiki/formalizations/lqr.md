---
type: formalization
tags: [control, lqr, optimal-control, linear-systems, locomotion]
status: complete
related:
  - ../concepts/quarter-car-active-suspension-lqr-optimization.md
  - ./eigenvalues-eigenvectors.md
  - ./kalman-filter.md
  - ../concepts/optimal-control.md
  - ../methods/model-predictive-control.md
  - ../methods/trajectory-optimization.md
sources:
  - ../../sources/sites/mit_underactuated_ch8_lqr_reading.md
  - ../../sources/courses/cmu_16_745_site_and_lecture_notes_reading_2025.md
  - ../../sources/papers/lqr_ilqr_primary_refs.md
  - ../../sources/papers/optimal_control_theory.md
  - ../../sources/papers/optimal_control.md
  - ../../sources/courses/mit_underactuated_kalman_lqr.md
summary: "LQR 给出线性系统二次代价下的最优状态反馈（Riccati 方程），可用打靶 / QP / DP 三种方式推导；iLQR 是其对非线性轨迹优化的迭代推广。"
updated: 2026-11-07
original:
  - ../../sources/sites/mit_underactuated_ch8_lqr_reading.md
  - ../../sources/courses/cmu_16_745_site_and_lecture_notes_reading_2025.md
---

# LQR / iLQR

**LQR（Linear Quadratic Regulator，线性二次调节器）**：最优控制中最经典的解析解，针对线性系统 + 二次代价函数，给出最优状态反馈增益的闭式解。**iLQR（iterative LQR）**是其非线性扩展，通过迭代线性化求解非线性轨迹优化。

## 一句话定义

> LQR 回答的是："对于一个线性系统，如果目标是最小化状态偏差和控制代价，最优的反馈控制律是什么？" iLQR 把这个框架推广到非线性系统，是轨迹优化的核心算法之一。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LQR | Linear Quadratic Regulator | 线性系统二次型代价下的最优反馈控制器 |
| iLQR | iterative Linear Quadratic Regulator | 对非线性系统迭代线性化求解的轨迹优化方法 |
| OCP | Optimal Control Problem | MPC 每步求解的有限时域最优控制问题 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |

## 为什么重要

- LQR 是最优控制（OCP）的最简洁、最完备的例子：有解析解，有稳定性保证
- 理解 LQR 是理解 MPC、Riccati 方程、Bellman 最优方程的基础
- iLQR 是 DDP（Differential Dynamic Programming）的简化版，是 trajectory optimization 的主流算法之一，Crocoddyl 用的就是 iLQR/FDDP
- 在人形控制中：LQR 用于平衡稳定，iLQR 用于轨迹优化

## 核心原理：LQR 问题

### 系统模型

线性时不变系统：

$$x_{t+1} = A x_t + B u_t$$

或连续时间版本：

$$\dot{x} = Ax + Bu$$

其中 $x$ 是状态向量、$u$ 是控制输入；$A$ 是**状态转移矩阵**，刻画系统在无控时的自然演化；$B$ 是**控制输入矩阵**，将控制量映射到状态空间。

### 代价函数

$$J = \sum_{t=0}^{T} \left( x_t^T Q x_t + u_t^T R u_t \right) + x_T^T Q_f x_T$$

其中：
- $Q \succeq 0$：状态代价矩阵（惩罚状态偏差）
- $R \succ 0$：控制代价矩阵（惩罚控制量大小）
- $Q_f$：终端状态代价

### 最优解：Riccati 方程

最优控制律是线性状态反馈：

$$u_t^* = -K_t x_t$$

其中增益矩阵 $K_t$ 由 **Riccati 方程** 反向递推求解：

$$P_t = Q + A^T P_{t+1} A - A^T P_{t+1} B (R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A$$

$$K_t = (R + B^T P_{t+1} B)^{-1} B^T P_{t+1} A$$

对于无限时域 LQR，Riccati 方程有稳态解 $P_\infty$（DARE 的正定不动点），对应时不变增益 $K_\infty$；数值上用 `dare` / `dlqr` / Drake `LinearQuadraticRegulator` 求。连续时间对应代数 Riccati 方程 $0=SA+A^\top S-SBR^{-1}B^\top S+Q$，$u^*=-R^{-1}B^\top Sx$，由 HJB 在猜 $J^*=x^\top Sx$ 后得到（Tedrake, Underactuated Ch.8）。

### LQR 的三种推导（CMU 16-745 第 7–9 讲；结果完全一致）

1. **间接打靶（Pontryagin）**：前向 rollout → 反向求协态 $\lambda$ → 线搜索更新控制，$u_k=-R^{-1}B^\top\lambda_{k+1}$；课程指出这类间接 / 打靶法随算力提升已大多不再使用。
2. **QP**：把整条轨迹堆成 $\min z^\top Hz$ s.t. $Cz=d$，KKT 系统极稀疏，一次线性求解即得精确解，远好于打靶；复杂度 $O((N(n+m))^3)$。
3. **动态规划 / Riccati**：值函数 $V_k(x)=\tfrac12x^\top P_kx$，反向递推 $P_N=Q_N$ 与上式；复杂度 $O(N(n+m)^3)$；与 QP 精确匹配。附带结论：**动力学约束的拉格朗日乘子就是值函数梯度** $\lambda_k=\nabla V_k(x_k)$（对非线性问题同样成立）。

可控性矩阵 $[B,AB,\dots,A^{n-1}B]$ 满秩（凯莱–哈密顿定理保证 $n$ 步足够）是 LQR 可行的简单条件；反馈形式使其能处理初值变化与扰动。

### 稳定性保证

若 $(A, B)$ 可镇定、$(A, \sqrt{Q})$ 可检测（可控 / 可观是更强的充分条件），则无限时域 LQR 有唯一的半正定稳态解且闭环渐近稳定；Tedrake 指出 Riccati 方程另有非正定解，对应不稳定控制器，拟合值迭代的稳定不动点只有正定解。闭环 $A-BK$ 的特征值决定模态衰减速度（见 [特征值与特征向量](./eigenvalues-eigenvectors.md)）。

## iLQR 问题

### 扩展到非线性系统

$$x_{t+1} = f(x_t, u_t)$$

$$J = \sum_{t=0}^{T} l(x_t, u_t) + l_f(x_T)$$

### 算法思路（Differential Dynamic Programming 简化版）

iLQR 是迭代算法，每轮包含：

**Backward pass（逆向传播）**：

1. 在当前轨迹 $\{x_t^*, u_t^*\}$ 处将系统线性化：$A_t = \partial f / \partial x$，$B_t = \partial f / \partial u$
2. 将代价函数二阶展开
3. 用 LQR 递推求解局部最优更新方向 $\delta u_t = K_t \delta x_t + k_t$

**Forward pass（正向传播）**：

用更新后的控制律 $u_t \leftarrow u_t^* + \alpha (K_t \delta x_t + k_t)$ 滚出新轨迹，更新参考轨迹

**迭代**直到收敛（代价不再下降）。

### 沿轨迹与非线性系统中的用法

- **局部镇定**：在平衡点 $(x_0,u_0)$ 线性化，$u=u_0-K(x-x_0)$。
- **时变 LQR（TVLQR）**：沿标称轨迹线性化得 $A(t),B(t)$，$u=u_0(t)-K(t)(x-x_0(t))$，在整条可行轨迹上有效；要谈轨迹稳定，轨迹须终止于可镇定不动点并用 $S_\infty$ 作终端条件。
- **四元数姿态**：朴素线性化含四元数状态的系统会得到不可控的线性系统，需用姿态雅可比把状态约化到 3 维误差（CMU 第 16 讲）；类似地，等式约束 $Fx=0$ 可在零空间基 $P$ 上做约化 LQR（Tedrake）。

### 优势与局限

| 属性 | LQR | iLQR |
|------|-----|-------|
| 系统假设 | 线性 | 非线性（可微） |
| 求解复杂度 | Riccati 总计 $O(N(n+m)^3)$ | 每次迭代约一次 backward + 一次 forward，$O(N\cdot\mathrm{poly}(n,m))$ |
| 全局最优 | ✅（线性系统） | ❌（局部最优） |
| 处理约束 | 不含不等式约束（加约束后需解 QP 即 MPC） | 本身不自然处理约束，接触等需靠前向仿真隐式满足或另加处理（CMU 第 20 讲） |
| 实时性 | 离线增益 → 在线线性反馈 | 需要迭代，不直接在线 |

## 工程实践：在机器人中的应用

### LQR 的典型用途
- 倒立摆稳定（入门 demo）
- 人形机器人躯干姿态稳定
- 线性化工作点附近的局部稳定控制
- 作为 MPC 的终端代价（terminal cost in MPC）

### iLQR 的典型用途
- 轨迹优化（连续接触动力学轨迹规划）
- Crocoddyl 中的 SolverFDDP / SolverBoxFDDP
- 全身运动生成（WBC 轨迹优化层）
- 人形行走摆腿轨迹设计

## 局限与风险

- 线性化只在工作点 / 轨迹附近有效；强非线性、大姿态误差需 TVLQR、四元数处理或改用 MPC / 轨迹优化。
- 标准 LQR 无输入 / 状态不等式约束；约束一加就没有闭式策略，转为每步解凸 QP（MPC）。
- $Q,R$ 权重需要人工调节，系统可镇定性不满足时求解会失败（例：未去掉被动或约束自由度的 Segway 全坐标模型，Tedrake）。
- iLQR 是局部方法，依赖初值；不直接处理不等式约束；Riccati 数值积分在弱可镇定系统上可能丢失对称 / 正定性（可积分平方根形式）。
- 不含噪声时的 LQR 与 LQG 的确定性等价、分离原理仅对线性高斯情形严格成立（CMU 第 19 讲）。

## LQR vs MPC 的关系

| 维度 | LQR | MPC |
|------|-----|-----|
| 预测窗口 | 无限时域（解析） | 有限时域（在线求解） |
| 处理约束 | 不直接支持 | 核心优势 |
| 非线性 | 需要线性化 | 可以处理非线性（NMPC） |
| 计算方式 | 离线 Riccati | 在线优化 |

MPC 在某种意义上是"在线、有约束、有限时域的 LQR"。CMU 第 10 讲的说法更具体：凸 MPC 以 LQR 的 cost-to-go 作终端代价，无约束时对任意时域与 LQR 完全一致；前 $H$ 步显式约束优化把状态带到约束不再起作用的区域，之后由 LQR cost-to-go 接管。

## 关联页面
- [1/4 车主动悬架：LQR 权重与带约束增益优化案例](../concepts/quarter-car-active-suspension-lqr-optimization.md) — 手调权重几乎无效、优化器只最优写下的约束（作者实操，未复现）

- [Optimal Control (OCP)](../concepts/optimal-control.md) — LQR 是 OCP 的最经典闭式解
- [Trajectory Optimization](../methods/trajectory-optimization.md) — iLQR 是 TO 的核心算法
- [Model Predictive Control (MPC)](../methods/model-predictive-control.md) — MPC 是 LQR 的有限时域、在线、带约束版本
- [Bellman 方程](./bellman-equation.md) — Riccati 方程是 Bellman 最优方程在线性二次问题上的解析形式
- [Lyapunov 稳定性](./lyapunov.md) — LQR 的闭环稳定性可通过 Lyapunov 视角理解
- [Crocoddyl](../entities/crocoddyl.md) — iLQR/FDDP 的开源实现

## 参考来源

- **原文通读归档：** [Tedrake *Underactuated Robotics* Ch.8 LQR 通读](../../sources/sites/mit_underactuated_ch8_lqr_reading.md)；[CMU 16-745 课程站与讲义通读](../../sources/courses/cmu_16_745_site_and_lecture_notes_reading_2025.md)（2026-11-07）。Kalman 1960/1964、Bryson & Ho、Li & Todorov 2004 等经典原论文未取得，下列索引为二手。
- [lqr_ilqr_primary_refs.md](../../sources/papers/lqr_ilqr_primary_refs.md) — LQR / iLQR 一手索引（Kalman 1964；Bryson & Ho 1975；Li & Todorov 2004 等）
- [mit_underactuated_kalman_lqr.md](../../sources/courses/mit_underactuated_kalman_lqr.md) — MIT LQR 三视角与 DDP 课程
- [optimal_control.md](../../sources/papers/optimal_control.md) — DDP / Bellman 等最优控制主线归档

## 推荐继续阅读

- Tedrake, *Underactuated Robotics* Ch.8 LQR <https://underactuated.csail.mit.edu/lqr.html>；CMU 16-745 讲义 <https://github.com/Optimal-Control-16-745/lecture-notebooks>
- [CMU Optimal Control 2025 策展](../entities/cmu-optimal-control-curriculum.md)
- [Optimal Control (OCP)](../concepts/optimal-control.md)
- [Trajectory Optimization](../methods/trajectory-optimization.md)
- [Crocoddyl](../entities/crocoddyl.md)（iLQR/FDDP 的工程实现）
- [HJB](./hjb.md) — LQR 是 HJB 方程在线性二次系统上的解析特例

## 一句话记忆

> LQR 是"线性系统最优反馈控制"的解析答案，Riccati 方程是它的核心工具；iLQR 是把这套框架推广到非线性系统的迭代算法，是轨迹优化的主力工具。
