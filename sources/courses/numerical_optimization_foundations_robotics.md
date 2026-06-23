# 数值优化基础（机器人应用）课程大纲

- **类型：** course
- **来源：** 具身智能研究室（微信公众号）课程大纲整理
- **收录日期：** 2026-06-23
- **一句话说明：** 面向机器人研究与工程的数值优化系统课程，从无约束/约束优化、对称锥规划到凸松弛与工程求解技巧。

## 为什么值得保留

- 把 **MPC / TrajOpt / WBC / 路径规划 / 接触距离** 背后共用的优化语言串成一条可执行学习线。
- 章节结构与 [运动控制成长路线](../../roadmap/motion-control.md) L3–L4 的「数值优化直觉 → 约束 QP → NMPC」高度对齐。
- 实践作业（线搜索、平滑导航路径、严格凸 QP、NMPC、PHR 增广拉格朗日、复杂障碍安全导航）可直接映射到本库方法页。

## 章节大纲（6 章）

### 第 1 章 数值优化基础

| 节 | 主题 |
|----|------|
| 1.1 | 数学规划与机器人 |
| 1.2 | 凸集与凸函数的高阶信息 |
| 1.3 | 凸函数性质 |
| 1.4 | 非凸无约束优化 — 线搜索最速下降 |
| 1.5 | 非凸无约束优化 — 修正阻尼牛顿法 |
| 1.6 | 实践：线搜索最速下降实现 |

### 第 2 章 无约束优化

| 节 | 主题 |
|----|------|
| 2.2 | 拟牛顿法（BFGS / L-BFGS） |
| 2.3 | 共轭梯度法 |
| 2.4 | 应用：平滑导航路径生成 |
| 2.5 | 实践：平滑导航路径 |

### 第 3 章 约束优化

| 节 | 主题 |
|----|------|
| 3.1 | 约束优化形式分类与复杂度 |
| 3.2 | 低维线性时间 LP — Seidel 算法 |
| 3.3 | 低维线性时间严格凸 QP |
| 3.4 | 三种序列无约束化方法（罚函数 / 障碍 / 等） |
| 3.5 | KKT 条件与 PHR 增广拉格朗日乘子法 |
| 3.6 | 应用 1：控制分配 |
| 3.7 | 应用 2：碰撞距离计算 |
| 3.8 | 应用 3：非线性模型预测控制（NMPC） |
| 3.9 | 实践：严格凸 QP + NMPC |

### 第 4 章 对称锥规划

| 节 | 主题 |
|----|------|
| 4.1 | 锥与对称锥 |
| 4.2 | 对称锥增广拉格朗日乘子法 |
| 4.3 | 应用：时间最优路径参数化（TOPP） |
| 4.4 | 实践：PHR 凸性证明与锥规划求解 |

### 第 5 章 优化问题构建与求解技巧

| 节 | 主题 |
|----|------|
| 5.1 | 函数光滑化技巧 |
| 5.2 | 伴随灵敏度分析 |
| 5.3 | 线性系统求解器分类与特性 |
| 5.4 | 优化软件资源 |
| 5.5 | 实践：复杂障碍环境安全导航 |

### 第 6 章 机器人学中的凸松弛

| 节 | 主题 |
|----|------|
| 6.2 | QCQP 凸松弛 |
| 6.3 | Riemannian Staircase 方法 |
| 6.4 | 分布式凸松弛 |
| 6.5 | GNC（Graduated Non-Convexity） |
| 6.6 | 推荐参考文献 |

## 对 wiki 的映射

| 课程主题 | wiki 页面 |
|---------|-----------|
| 课程总览 | [numerical-optimization-curriculum](../../wiki/entities/numerical-optimization-curriculum.md) |
| 凸函数 / 凸集 | [convex-functions](../../wiki/formalizations/convex-functions.md) |
| 线搜索最速下降 | [line-search-steepest-descent](../../wiki/methods/line-search-steepest-descent.md) |
| 拟牛顿 BFGS | [quasi-newton-bfgs](../../wiki/methods/quasi-newton-bfgs.md) |
| 共轭梯度 | [conjugate-gradient-method](../../wiki/methods/conjugate-gradient-method.md) |
| 平滑导航路径 | [smooth-navigation-path-generation](../../wiki/methods/smooth-navigation-path-generation.md) |
| 约束优化 | [constrained-optimization](../../wiki/concepts/constrained-optimization.md) |
| KKT | [kkt-conditions](../../wiki/formalizations/kkt-conditions.md) |
| QP | [quadratic-programming](../../wiki/formalizations/quadratic-programming.md) |
| 罚 / 障碍 / 增广拉格朗日 | [penalty-barrier-augmented-lagrangian](../../wiki/methods/penalty-barrier-augmented-lagrangian.md) |
| 控制分配 | [control-allocation](../../wiki/concepts/control-allocation.md) |
| 碰撞距离 | [collision-distance-optimization](../../wiki/concepts/collision-distance-optimization.md) |
| NMPC | [nonlinear-model-predictive-control](../../wiki/methods/nonlinear-model-predictive-control.md) |
| 对称锥规划 | [symmetric-cone-programming](../../wiki/formalizations/symmetric-cone-programming.md) |
| TOPP | [time-optimal-path-parameterization](../../wiki/methods/time-optimal-path-parameterization.md) |
| 伴随灵敏度 | [adjoint-sensitivity-analysis](../../wiki/formalizations/adjoint-sensitivity-analysis.md) |
| 优化软件选型 | [optimization-software-selection](../../wiki/queries/optimization-software-selection.md) |
| 凸松弛 / GNC / Riemannian | [convex-relaxation-robotics](../../wiki/methods/convex-relaxation-robotics.md) |
| 上游 OCP / MPC / TrajOpt | [optimal-control](../../wiki/concepts/optimal-control.md)、[model-predictive-control](../../wiki/methods/model-predictive-control.md)、[trajectory-optimization](../../wiki/methods/trajectory-optimization.md) |

## 推荐延伸阅读（课程 6.6 同类）

- Stephen Boyd & Lieven Vandenberghe, *Convex Optimization*
- Jorge Nocedal & Stephen Wright, *Numerical Optimization*
- Russ Tedrake, *Underactuated Robotics*（OCP / iLQR / MPC 机器人视角）
