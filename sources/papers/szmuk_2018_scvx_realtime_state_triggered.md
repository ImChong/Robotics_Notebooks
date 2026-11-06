# Successive Convexification for Real-Time 6-DoF Powered Descent Guidance with State-Triggered Constraints

> 来源归档（ingest）

- **标题：** Successive Convexification for Real-Time 6-DoF Powered Descent Guidance with State-Triggered Constraints
- **作者：** Michael Szmuk, Taylor P. Reynolds, Behçet Açıkmeşe（University of Washington）
- **论文：** <https://arxiv.org/abs/1811.10803>（v1，2018-11-27）
- **出版物：** arXiv 注释为 "Submitted to AIAA Journal of Guidance Navigation and Control"；arXiv 页列出相关 DOI 10.2514/1.G004549（正式发表卷期本次未核对）
- **类型：** journal paper 预印本
- **入库日期：** 2026-11-06
- **核对范围：** 用 pdftotext 抽取全文后通读 §I–V、表 1–4 与参考文献；图 1/3/4/6 文字错乱，个别符号（如 ṁ、≠）丢失。
- **通读：** 已完成
- **一句话说明：** 在 6-DoF SCvx 着陆制导上加入自由点火时间、升阻气动模型和"状态触发约束"，MATLAB+ECOS 下单次求解最长 0.682 s。

## 核心摘录

1. **三项贡献（摘要）。** "(i) a free-ignition-time modification that allows the algorithm to determine the optimal engine ignition time, (ii) a tractable aerodynamics formulation that models both lift and drag, and (iii) a continuous state-triggered constraint formulation that emulates conditionally enforced constraints."
2. **状态触发约束（§II.F，式 15–19）。** 语义是 "g(z) < 0 ⇒ c(z) = 0"（只单向蕴含，区别于双向的互补约束）；借线性互补问题的解析解 \(\hat\sigma=-\min(g,0)\) 化成连续约束 \(h(z)=-\min(g(z),0)\cdot c(z)=0\)。例：速度超过阈值才限制攻角、距离足够近才施加视线约束。
3. **气动（§II.E，式 14）。** \(A_B=-\tfrac12\rho V S_A C_A v\)，\(C_A\) 为正定对角阵；轴向与横向系数相等时退化为纯阻力，不等时产生升力。密度取常值、压心固定。
4. **自由点火时间（§II.D）。** 把无动力弹道段的滑行时间 \(t_c\) 设为决策变量，点火时的位置、速度用滑行多项式表示；算例中求得 \(t_c=0.96\) UT，燃烧时间从 4.67 降到 2.56 UT。
5. **实时性（§IV.D，表 4）。** 2014 MacBook Pro（2.2 GHz i7）上用 MATLAB + ECOS + CVX 求解凸子问题，传播步用 C++/Eigen（约 10 ms，未计入表 4）；K = 20，4–7 次迭代收敛，最长 0.682 s。§III.C.2 明说 "no convergence guarantees are presented"。
6. **飞行验证。** 论文没有飞行试验，结论写道在代表性飞行硬件上的结果 "will be crucial"。

## 对 wiki 的映射

- [序列凸化（Successive Convexification, SCvx）](../../wiki/methods/successive-convexification.md)
- [火箭回收控制纵深路线](../../roadmap/depth-rocket-landing.md) — Stage 3 / Stage 5
