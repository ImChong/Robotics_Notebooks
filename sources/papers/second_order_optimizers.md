# second_order_optimizers

> 来源归档（ingest）

- **标题：** Second-Order & Quasi-Newton Optimizers — Newton / Gauss-Newton / Levenberg-Marquardt / BFGS / L-BFGS / Truncated Newton
- **类型：** paper + textbook + course
- **来源：** 经典数值优化论文、Nocedal & Wright、机器人 TrajOpt 课程
- **入库日期：** 2026-06-27
- **一句话说明：** 覆盖机器人 TrajOpt、IK、状态估计与非线性最小二乘中常见的二阶与拟牛顿优化器一手出处，支撑各方法独立 method 页与选型对比。

## 核心论文摘录（MVP）

### 1) Kantorovich 定理与牛顿法局部二次收敛（Kantorovich, 1948；现代表述见 Dennis & Schnabel, 1983）
- **链接：** Dennis & Schnabel, *Numerical Methods for Unconstrained Optimization and Nonlinear Equations* — <https://doi.org/10.1137/1.9781611971200>
- **核心贡献：** **牛顿法（Newton's method）**：用 Hessian $\nabla^2 f(x_k)$ 构造二次模型，搜索方向 $p_k = -(\nabla^2 f)^{-1}\nabla f$，在强凸邻域内二次收敛；非凸时需阻尼或修正保证下降。
- **关键更新：** $x_{k+1} = x_k - \alpha_k (\nabla^2 f(x_k))^{-1} \nabla f(x_k)$
- **对 wiki 的映射：**
  - [Newton's Method](../../wiki/methods/newtons-method.md)
  - [Second-Order Optimizers 对比](../../wiki/comparisons/second-order-optimizers.md)

### 2) Theoria Motus Corporum Coelestium（Gauss, 1809）/ 非线性最小二乘现代表述（Gill, Murray & Wright）
- **链接：** Nocedal & Wright, *Numerical Optimization* Ch. 10 — <https://doi.org/10.1007/978-0-387-40065-5>
- **核心贡献：** **Gauss-Newton**：对残差 $r(x)$ 最小化 $\|r(x)\|^2$，用 Jacobian $J$ 近似 Hessian 为 $J^T J$，避免显式二阶导；是 TrajOpt 打靶、IK、标定等最小二乘问题的默认曲率模型。
- **关键方向：** $p = -(J^T J)^{-1} J^T r$
- **对 wiki 的映射：**
  - [Gauss-Newton](../../wiki/methods/gauss-newton.md)
  - [Trajectory Optimization](../../wiki/methods/trajectory-optimization.md)

### 3) An Algorithm for Least-Squares Estimation of Nonlinear Parameters (Levenberg, 1944)
- **链接：** <https://doi.org/10.1137/0116009>
- **核心贡献：** 在 Gauss-Newton 方向上加阻尼 $\lambda I$，在梯度下降与 GN 之间插值，改善病态 Jacobian 时的稳定性。
- **对 wiki 的映射：**
  - [Levenberg-Marquardt](../../wiki/methods/levenberg-marquardt.md)

### 4) An Algorithm for Least-Squares Estimation of Nonlinear Parameters (Marquardt, 1963)
- **链接：** <https://doi.org/10.1137/0116030>
- **核心贡献：** 独立提出与 Levenberg 等价的阻尼策略；**Levenberg-Marquardt（LM）** 成为非线性最小二乘（相机标定、状态估计、曲线拟合）的事实标准。
- **对 wiki 的映射：**
  - [Levenberg-Marquardt](../../wiki/methods/levenberg-marquardt.md)

### 5) A Family of Variable-Metric Methods Derived by Variational Means (Broyden, 1970) 等 BFGS 四篇
- **链接：**
  - Broyden (1970): <https://doi.org/10.1090/S0002-9947-1970-0258249-9>
  - Fletcher (1970): <https://doi.org/10.1093/comjnl/13.4.317>
  - Goldfarb (1970): <https://doi.org/10.1007/BF01585369>
  - Shanno (1970): <https://doi.org/10.1007/BF01588908>
- **核心贡献：** **BFGS 拟牛顿更新**：用梯度差分 $s_k, y_k$ 低秩修正近似 Hessian 逆 $H_k$，每步 $O(n^2)$ 存储，凸光滑问题超线性收敛。
- **对 wiki 的映射：**
  - [BFGS](../../wiki/methods/bfgs.md)

### 6) On the Limited Memory BFGS Method for Large Scale Optimization (Liu & Nocedal, 1989)
- **链接：** <https://doi.org/10.1007/BF01582236>
- **核心贡献：** **L-BFGS**：只存最近 $m$ 对 $(s,y)$，用 two-loop recursion 计算搜索方向，内存 $O(mn)$；高维 TrajOpt（如 [cuRobo](../../wiki/entities/curobo.md)）工业默认。
- **对 wiki 的映射：**
  - [L-BFGS](../../wiki/methods/l-bfgs.md)
  - [Quasi-Newton BFGS（总览）](../../wiki/methods/quasi-newton-bfgs.md)

### 7) A Survey of Truncated-Newton Methods (Nash, 2000)
- **链接：** <https://doi.org/10.1137/S0036144500377774>
- **核心贡献：** **截断牛顿 / Newton-CG**：每步用 [共轭梯度](../../wiki/methods/conjugate-gradient-method.md) 近似求解 Newton 方程 $Hp = -g$，early stop 控制计算预算；适合大型稀疏 Hessian 或 Hessian-vector product 可得的 NLP。
- **对 wiki 的映射：**
  - [Truncated Newton](../../wiki/methods/truncated-newton.md)

## 辅助一手资料

- [数值优化基础（机器人应用）课程](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 1 章阻尼牛顿、第 2 章 BFGS/CG
- Nocedal & Wright, *Numerical Optimization* (2nd ed.) — Ch 2（线搜索）、Ch 6（拟牛顿）、Ch 7（大规模无约束）、Ch 10（最小二乘）

## 当前提炼状态

- [x] 6 类常见二阶/拟牛顿优化器各有一条以上一手出处
- [x] 映射到独立 wiki method 页与对比页
- [ ] 后续补：与一阶 Adam/SGD 在机器人学习栈中的分层对照
