# OSQP：基于算子分裂的二次规划求解器（通读）

> 来源归档（原文通读 · 支撑 `wiki/queries/mpc-solver-selection.md`、`wiki/queries/wbc-implementation-guide.md`、`wiki/queries/wbc-tuning-guide.md`）

- **标题：** OSQP: An Operator Splitting Solver for Quadratic Programs
- **作者：** Bartolomeo Stellato, Goran Banjac, Paul Goulart, Alberto Bemporad, Stephen Boyd
- **类型：** paper
- **链接：** <https://arxiv.org/abs/1711.08013>（PDF：<https://arxiv.org/pdf/1711.08013>）；Mathematical Programming Computation 12, 2020
- **版本 / 日期：** arXiv v4，2020-02-12
- **入库日期：** 2026-11-07
- **核对范围：** 通读摘要、§1（问题与三类求解方法综述）、§2–§5 算法（算子分裂、拟定性线性系统、求解 polishing、ρ 自适应规则）、§6 参数化问题（因子分解缓存与 warm start）、§7 实现、§8 基准与结论；附录（各问题类的随机生成方式、部分证明）与参考文献仅浏览，未逐式核对证明。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题形式：** $\min \tfrac12x^TPx+q^Tx$ s.t. $l\le Ax\le u$，$P\succeq0$（凸 QP；令 $l_i=u_i$ 表示等式）。
- **方法谱系（§1.2）：** 有效集法易 warm start，但最坏情形复杂度随约束数指数增长（开源代表 qpOASES）；内点法多项式复杂度、精度高，但不易 warm start 且大规模扩展差；一阶法（ADMM）每步便宜、适合嵌入式，但迭代数依赖数据与步长参数，通常只给中等精度。
- **OSQP 的做法：** ADMM + 新的算子分裂，每次迭代求解**同一个拟定性（quasi-definite）线性系统**，对问题数据无严格凸 / 线性无关假设；固定 ρ 时只做一次矩阵分解，之后可 division-free；首个能从迭代序列可靠检测原 / 对偶不可行的算子分裂 QP 法；可选 **solution polishing**（猜出有效约束后解一个小的等式约束 QP，在识别正确时精度不低于内点法）。
- **ρ 自适应（§5）：** 用原始 / 对偶残差比的平方根更新 ρ，初值 $\bar\rho_0=0.1$，基准中通常只更新 1–2 次；仅当累计迭代时间超过分解时间的约 40% 且新值与旧值相差 5 倍以上才更新（更新需重新数值分解）。
- **参数化问题（§6）：** 仅向量 $q,l,u$ 变化时 KKT 矩阵不变 → 缓存分解；矩阵数值变化但稀疏结构不变 → 只重做数值分解；**warm start 用上一次解** $(x^*,Ax^*,y^*)$ 初始化。线性 MPC / MHE 属于第一类。
- **基准（§8）：** 7 类问题 × 20 个规模 × 10 个实例共 1400 个，$n\in[10^1,10^4]$、$m\in[10^2,10^5]$；对比 ECOS、GUROBI、MOSEK（未含 qpOASES）。结论：OSQP 在这些基准上总体最好（MOSEK 低精度更好，GUROBI 高精度更好）；摘要称典型情形比内点法快约 10 倍；warm start + 分解缓存在 Lasso 路径上带来 10 倍以上提速、迭代数减少 8–11 倍；MPC 例子（$n_x$=20–80，$T$=10，100 步仿真）同样受益。
- **局限：** 默认精度中等（实验用 $10^{-3}$）；病态 / 退化问题上 polishing 很少成功；ρ 更新频繁时收敛性难证明（原文只给了"更新有限次则收敛"）。

## 对 wiki 的映射

- [`wiki/queries/mpc-solver-selection.md`](../../wiki/queries/mpc-solver-selection.md)、[`wiki/queries/wbc-implementation-guide.md`](../../wiki/queries/wbc-implementation-guide.md)、[`wiki/queries/wbc-tuning-guide.md`](../../wiki/queries/wbc-tuning-guide.md)

## 参考来源

- Stellato, B., Banjac, G., Goulart, P., Bemporad, A., Boyd, S. (2020). OSQP: An operator splitting solver for quadratic programs. *Math. Prog. Comp.* 12. <https://arxiv.org/abs/1711.08013>
