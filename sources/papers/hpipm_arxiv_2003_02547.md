# HPIPM：面向 MPC 的高性能二次规划框架（通读）

> 来源归档（原文通读 · 支撑 `wiki/queries/mpc-solver-selection.md`）

- **标题：** HPIPM: a high-performance quadratic programming framework for model predictive control
- **作者：** Gianluca Frison, Moritz Diehl
- **类型：** paper
- **链接：** <https://arxiv.org/abs/2003.02547>（PDF：<https://arxiv.org/pdf/2003.02547>）；IFAC World Congress 2020
- **版本 / 日期：** arXiv v2，2020-06-06
- **入库日期：** 2026-11-07
- **核对范围：** 通读全文 §1–§6（QP 类型、IPM 算法、delta / absolute 公式、KKT 求解与迭代细化、凝聚、实现选择、三组数值实验与结论）；参考文献仅浏览。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **内容：** 开源 C 语言 QP 框架，三种 QP 类型（稠密、OCP、树结构 OCP），Mehrotra 型内点法（speed_abs / speed / balance / robust 多种模式，速度与鲁棒性取舍），以及（部分）凝聚例程；取代 HPMPC，线性代数基于 BLASFEO。
- **实验：** ①线性质量弹簧（M=2..15 质量、N=10..30 阶段，固定 10 次 IPM 迭代）：speed_abs 与 HPMPC 差 10–15%，speed 小规模约慢 2 倍，robust 约慢 3 倍。②Maros-Mészáros 43 个小规模凸 QP（≤250 变量、≤1001 约束）：speed 模式约 50% 题目最快但有 4 题失败，balance 全部解出并在 >90% 题目中不超过最快者的 2 倍，qpOASES 约 30% 题目最快但在其余题目上常慢 10 倍以上。③acados RTI 生成的链式质量块 MPC QP（24 状态、3 控制、40 阶段、50 步闭环）：**HPIPM 解部分凝聚后的 OCP QP 最快，稳定低于 1 ms**；HPIPM（内点法）耗时几乎恒定，qpOASES（有效集）与 OSQP（ADMM）在问题难的时段迭代更多、耗时更长；HPIPM 与 OSQP 解部分凝聚 QP 都显著快于完全稀疏版本。
- **局限：** 速度模式鲁棒性较差；基准机器为 Core i7 4810MQ 的单线程，结论不外推到别的硬件 / 问题。

## 对 wiki 的映射

- [`wiki/queries/mpc-solver-selection.md`](../../wiki/queries/mpc-solver-selection.md)

## 参考来源

- Frison, G., Diehl, M. (2020). HPIPM: a high-performance quadratic programming framework for model predictive control. IFAC-PapersOnLine 53(2). <https://arxiv.org/abs/2003.02547>
