# optimal_control

> 来源归档（ingest）

- **标题：** Optimal Control / Dynamic Programming / Trajectory Optimization
- **类型：** paper / textbook
- **来源：** 教材 / 经典论文
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-14
- **一句话说明：** 最优控制理论基础，覆盖 Bellman 动态规划、Pontryagin 极大值原理和 LQR/iLQR，支撑 OCP、LQR、轨迹优化等 wiki 页面。

## 核心论文摘录（MVP）

### 1) Dynamic Programming (Bellman, 1957)
- **来源：** Princeton University Press
- **核心贡献：** 提出动态规划原理（最优子结构 + 无后效性），建立了 Value Function 与 Bellman 方程的理论基础，是现代最优控制与强化学习的共同源头。
- **关键结论：** 最优策略满足 $V^*(x) = \min_u \left[ l(x,u) + V^*(f(x,u)) \right]$
- **对 wiki 的映射：**
  - [Optimal Control (OCP)](../../wiki/concepts/optimal-control.md)
  - [Bellman 方程](../../wiki/formalizations/bellman-equation.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

### 2) A Pontryagin Minimum Principle (Pontryagin et al., 1962)
- **来源：** Interscience Publishers
- **核心贡献：** 给出连续时间系统的最优控制必要条件（Hamiltonian 极小化），比 DP 更适合连续动作空间，是轨迹优化和运动规划的数学基础。
- **关键公式：** $H(x^*, u^*, \lambda^*) \leq H(x^*, u, \lambda^*)\ \forall u \in U$
- **对 wiki 的映射：**
  - [Optimal Control (OCP)](../../wiki/concepts/optimal-control.md)
  - [Trajectory Optimization](../../wiki/methods/trajectory-optimization.md)

### 3) Applied Optimal Control (Bryson & Ho, 1975)
- **来源：** Taylor & Francis
- **核心贡献：** 将最优控制理论系统化为工程可用的方法集合，包括 LQR、LQG、共轭梯度法等，是机器人控制工程师的标准参考。
- **对 wiki 的映射：**
  - [LQR / iLQR](../../wiki/formalizations/lqr.md)
  - [Lyapunov 稳定性](../../wiki/formalizations/lyapunov.md)
  - [Optimal Control (OCP)](../../wiki/concepts/optimal-control.md)

### 4) Differential Dynamic Programming (Mayne, 1966 / Jacobson & Mayne, 1970)
- **来源：** IEEE TAC / Elsevier
- **核心贡献：** DDP 算法：通过二阶 Taylor 展开在轨迹附近局部优化，是 iLQR 的前身，Crocoddyl 等工具的理论基础。
- **对 wiki 的映射：**
  - [LQR / iLQR](../../wiki/formalizations/lqr.md)
  - [Trajectory Optimization](../../wiki/methods/trajectory-optimization.md)

### 5) Reinforcement Learning and Optimal Control (Bertsekas, 2019)
- **来源：** Athena Scientific
- **核心贡献：** 系统梳理 RL 与最优控制的统一框架，从 DP 出发推导 TD、Q-learning、策略梯度，是两个领域的理论桥梁。
- **对 wiki 的映射：**
  - [Bellman 方程](../../wiki/formalizations/bellman-equation.md)
  - [MDP](../../wiki/formalizations/mdp.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

## 当前提炼状态

- [x] Bellman DP / Pontryagin / Bryson&Ho / DDP / Bertsekas 五条主线摘要
- [ ] 后续补：离散 vs 连续时间最优控制对比表
- [ ] 后续补：在机器人中的典型应用（Crocoddyl FDDP、ALTRO、IPOPT）
