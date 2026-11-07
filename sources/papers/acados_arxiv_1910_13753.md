# acados：模块化嵌入式快速最优控制框架（通读）

> 来源归档（原文通读 · 支撑 `wiki/queries/mpc-solver-selection.md`）

- **标题：** acados – a modular open-source framework for fast embedded optimal control
- **作者：** Robin Verschueren, Gianluca Frison, Dimitris Kouzoupis, Jonathan Frey, Niels van Duijkeren, Andrea Zanelli, Branimir Novoselnik, Thivaharan Albin, Rien Quirynen, Moritz Diehl
- **类型：** paper
- **链接：** <https://arxiv.org/abs/1910.13753>（PDF：<https://arxiv.org/pdf/1910.13753>）；Mathematical Programming Computation 14, 2022
- **版本 / 日期：** arXiv v3，2020-11-26
- **入库日期：** 2026-11-07
- **核对范围：** 通读摘要、§1 引言、§2 算法要素（2.1–2.8：OCP 形式、多重打靶、SQP 与实时迭代 RTI、积分与灵敏度、Gauss-Newton / SCQP / 凸化的 Hessian 近似、结构化 QP 求解、RTI）、§3（BLASFEO、HPIPM、AD、积分器、SQP 型方法）、§4（核心库、C 接口、高层接口）、§5 数值结果的案例 1 前半与案例 3 嵌入式实验、§6 结论；§5.2 Hessian 正则化的详细曲线、附录中的模型方程与参考文献仅浏览。未运行代码。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位：** 嵌入式 NMPC / MHE 求解器集合；核心库 C 实现、线性代数基于 BLASFEO，提供 Python / Matlab 接口并以 CasADi 为建模语言（摘要强调核心"不依赖自动代码生成"，高层接口再用 CasADi 代码生成函数并生成 C 工程）。BSD-2-Clause。
- **算法：** 多重打靶离散；SQP 型方法（Gauss-Newton、SCQP、带正则化 / 凸化的精确 Hessian）及 **RTI**（每个采样时刻只做一次完整 SQP 迭代）；**不含全局化**（线搜索 / 信赖域），理由是实时场景无法给出时间上界且初值通常靠近解。
- **QP 层：** HPIPM（Mehrotra 型原对偶内点法）提供稠密 / OCP / 树结构 QP 及（部分）凝聚；"另有接口"到 HPMPC、qpDUNES、qpOASES、OSQP（因此 acados 里 HPIPM 是默认 QP 框架）。2.7 节对比：稀疏法（CVXGEN / OOQP / OSQP）、结构化法（FORCES、HPMPC、HPIPM，均内点法）、凝聚法（配 qpOASES，凝聚的复杂度与时域长度呈二次关系）、部分凝聚、qpDUNES。
- **线性代数（§3.1）：** 对 10×10 到 100×100 的典型 MPC 规模，代码生成的线性代数不如优化库；BLASFEO 对小矩阵（≲300×300）比 OpenBLAS 等也有显著加速。
- **结果：** 链式质量块基准（§5.1，与 IPOPT、FalcOPT、VIATOC、ACADO、GRAMPC 对比）：acados 与 GRAMPC 在"次优性—计算时间"Pareto 前沿上，acados 次优性低约 1000 倍；中位计算时间 acados 约 1.05 ms vs ACADO 1.97 ms、GRAMPC 1.06 ms、IPOPT 59.84 ms；**未与 FORCES Pro 做数值对比**。案例 3：dSPACE MicroAutoboxII（900 MHz PowerPC）上，N=20 的发动机控制 NMPC 达到毫秒量级。
- **局限 / 展望：** 全局化策略、Simulink 互操作、非线性内点法等留待后续。

## 对 wiki 的映射

- [`wiki/queries/mpc-solver-selection.md`](../../wiki/queries/mpc-solver-selection.md)

## 参考来源

- Verschueren, R. et al. (2022). acados — a modular open-source framework for fast embedded optimal control. *Math. Prog. Comp.* 14. <https://arxiv.org/abs/1910.13753>
