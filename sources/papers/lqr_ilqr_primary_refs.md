# LQR / iLQR 一手资料索引

> 来源归档（ingest）

- **标题：** Linear Quadratic Regulator (LQR) 与 iterative LQR (iLQR) 经典论文、教材与课程
- **类型：** paper / textbook / course（合集）
- **入库日期：** 2026-06-01
- **一句话说明：** 汇总 LQR 解析解来源、Riccati 方程、DDP/iLQR 迭代轨迹优化原始文献与 MIT 最优控制课程，支撑 `wiki/formalizations/lqr.md` 与 `wiki/methods/lqr-ilqr.md`。
- **沉淀到 wiki：** 是 → [lqr](../../wiki/formalizations/lqr.md)、[lqr-ilqr](../../wiki/methods/lqr-ilqr.md)、[trajectory-optimization](../../wiki/methods/trajectory-optimization.md)

## 为什么值得保留

- **LQR** 是机器人平衡、MPC 终端代价、RL 中 model-based planning 的共同数学底座；**iLQR** 是 MuJoCo / Crocoddyl 轨迹优化的默认引擎。
- 一手资料需覆盖：**线性二次解析解 → DDP 二阶展开 → iLQR 简化 → 带约束扩展** 四条线，避免只记公式不知出处。

## 核心摘录

### 1) Kalman (1964) — 线性系统何时最优（LQR 理论支点）

- **来源：** R. E. Kalman, *When Is a Linear Control System Optimal?*, ASME Journal of Basic Engineering, 86:51–60, 1964.
- **要点：**
  - 给出 **线性反馈最优性** 的充要结构：闭环特征值、传递函数与 **Riccati 方程解** 的关系。
  - 连接 **滤波（KF）与控制（LQR）** 的对偶视角，是 LQG 设计的理论背景。
- **对 wiki 的映射：** [lqr](../../wiki/formalizations/lqr.md)、[bellman-equation](../../wiki/formalizations/bellman-equation.md)

### 2) Bryson & Ho (1975) — *Applied Optimal Control*

- **来源：** A. E. Bryson, Y.-C. Ho, *Applied Optimal Control: Optimization, Estimation, and Control*, Taylor & Francis, 1975.
- **要点：**
  - 工程向 **LQR、LQG、共轭梯度法** 与 **Pontryagin 极大值原理** 并列讲授。
  - 离散/连续 Riccati 递推、权重 $Q,R$ 调参、稳定性条件 $(A,B)$ 可控 / $(A,\sqrt{Q})$ 可观 —— 机器人课程最常引用的 **LQR 教材** 之一。
- **对 wiki 的映射：** [lqr](../../wiki/formalizations/lqr.md)、[optimal-control](../../wiki/concepts/optimal-control.md)；亦见 [optimal_control.md](./optimal_control.md)

### 3) Anderson & Moore (2007) — *Optimal Control: Linear Quadratic Methods* (2nd ed.)

- **来源：** B. D. O. Anderson, J. B. Moore, *Optimal Control: Linear Quadratic Methods*, Dover, 2007（Prentice-Hall 1990 修订版重印）.
- **要点：**
  - **LQR / LQG / H₂** 的严格处理：无限时域 $P_\infty$、谱分解、鲁棒性备注。
  - 适合需要 **定理级稳定性证明** 的读者，与 Bryson & Ho 的工程推导互补。
- **对 wiki 的映射：** [lqr](../../wiki/formalizations/lqr.md)、[lyapunov](../../wiki/formalizations/lyapunov.md)

### 4) Jacobson & Mayne (1970) — *Differential Dynamic Programming*（DDP，iLQR 前身）

- **来源：** D. H. Jacobson, D. Q. Mayne, *Differential Dynamic Programming*, Elsevier, 1970.
- **要点：**
  - 在参考轨迹上对动力学做 **二阶展开**、对代价做 **二阶展开**，反向 Riccati 型递推得到 **Newton 型** 控制更新。
  - **iLQR** 常被视为忽略某些二阶交叉项的 DDP 特例；Crocoddyl 的 FDDP 更接近完整 DDP。
- **对 wiki 的映射：** [lqr-ilqr](../../wiki/methods/lqr-ilqr.md)、[trajectory-optimization](../../wiki/methods/trajectory-optimization.md)、[optimal_control.md](./optimal_control.md) §DDP

### 5) Li & Todorov (2004) — iLQR 命名与生物运动控制

- **来源：** W. Li, E. Todorov, *Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems*, ICINCO, 2004. [作者 PDF](https://homes.cs.washington.edu/~todorov/papers/LiTodorov04.pdf)
- **要点：**
  - 提出 **iterative LQR**：每轮在当前轨迹线性化动力学、二次近似代价，解 **时变 LQR** 得到 $\delta u$ 并线搜索更新。
  - 面向 **高自由度人体 / 机器人运动**；计算量低于完整 DDP，成为后续机器人库默认算法名。
- **对 wiki 的映射：** [lqr-ilqr](../../wiki/methods/lqr-ilqr.md)、[lqr](../../wiki/formalizations/lqr.md)

### 6) Todorov & Li (2005) — 广义 iLQG（随机、采样系统）

- **来源：** E. Todorov, W. Li, *A Generalized Iterative LQG Method for Locally-Optimal Feedback Control of Constrained Nonlinear Stochastic Systems*, ACC, 2005. [PDF](https://homes.cs.washington.edu/~todorov/papers/TodorovLi05.pdf)
- **要点：**
  - 将 iLQR 推广到 **带噪声（LQG 型）** 与 **控制限幅** 的局部最优反馈；统一 **iLQG** 命名。
  - 强调 **局部收敛** 与 **轨迹邻域** 有效性 —— 机器人轨迹优化必须配合 **热启动 / 正则化**。
- **对 wiki 的映射：** [lqr](../../wiki/formalizations/lqr.md)、[lqr-ilqr](../../wiki/methods/lqr-ilqr.md)

### 7) Tassa et al. (2012, 2014) — MuJoCo 在线轨迹优化与约束 iLQR

- **来源：**
  - Y. Tassa et al., *Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization*, IROS, 2012.
  - Y. Tassa, N. Mansard, E. Todorov, *Control-Limited Differential Dynamic Programming*, ICRA, 2014. [PDF](https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf)
- **要点：**
  - 2012：在 **MuJoCo** 上展示 **模型预测式 iLQR** 合成复杂操作行为。
  - 2014：**控制限幅** 下的 DDP/iLQR 修正（Box 约束 backward pass），影响后续 Box-iLQR / MPC 实现。
- **对 wiki 的映射：** [lqr-ilqr](../../wiki/methods/lqr-ilqr.md)、[mujoco](../../wiki/entities/mujoco.md)、[model-predictive-control](../../wiki/methods/model-predictive-control.md)

### 8) MIT 6.832 / Optimal Control 2025 — LQR 三视角与 DDP 讲义

- **来源：** R. Tedrake, MIT [Optimal Control 2025 播放列表](https://www.youtube.com/playlist?list=PLZnJoM76RM6IAJfMXd1PgGNXn3dxhkVgI)（Lecture 8: LQR three ways；Lecture 12: DDP；Lecture 21: Kalman & duality）
- **要点：**
  - **LQR 三种推导**：动态规划 / 庞特里亚金 / 代数 Riccati。
  - **DDP ↔ iLQR** 与 **估计–控制对偶** 在同一课程闭环呈现。
- **对 wiki 的映射：** [lqr](../../wiki/formalizations/lqr.md)、[optimal-control](../../wiki/concepts/optimal-control.md)；课程归档见 [mit_underactuated_kalman_lqr](../courses/mit_underactuated_kalman_lqr.md)

## 推荐继续阅读（外部）

- Mayne (1966) 原始 DDP 论文 — 已摘要于 [optimal_control.md](./optimal_control.md)
- Bertsekas (2019) *Reinforcement Learning and Optimal Control* — DP 与 RL 统一（已收录 optimal_control.md）

## 当前提炼状态

- [x] LQR / iLQR 八类一手来源摘录与 wiki 映射
- [x] 回写 `wiki/formalizations/lqr.md` 与 `wiki/methods/lqr-ilqr.md` 的 `## 参考来源`
- [ ] 后续可补：连续时间 LQR、离散化与 ZOH 对照表
