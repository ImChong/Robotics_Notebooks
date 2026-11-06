# Minimum-Landing-Error Powered-Descent Guidance for Mars Landing Using Convex Optimization

> 来源归档（ingest）

- **标题：** Minimum-Landing-Error Powered-Descent Guidance for Mars Landing Using Convex Optimization
- **作者：** Lars Blackmore, Behçet Açıkmeşe, Daniel P. Scharf（JPL / Caltech）
- **出版物：** *Journal of Guidance, Control, and Dynamics*, Vol. 33, No. 4, July–August 2010, pp. 1161–1171，DOI: 10.2514/1.47202
- **类型：** journal paper
- **可读副本：** <https://ece.uwaterloo.ca/~ece602/Projects/2016/Papers/paper9.pdf>（滑铁卢大学课程项目目录中的期刊版 PDF）
- **入库日期：** 2026-11-06
- **核对范围：** 用 pdftotext 抽取全文后通读 §I–VI、附录（引理 4–6）与参考文献 [1]–[27]。抽取文本丢失了希腊字母和部分运算符，式 (48)(62)(66) 版面错乱，下文式 (48) 的符号按标准形式重建，未逐符号对照原版 PDF。
- **通读：** 已完成
- **一句话说明：** 把带最小推力约束的非凸火星动力下降问题无损松弛成 SOCP，并在目标点不可达时先最小化着陆误差、再在该误差下最省燃料。

## 核心摘录

1. **问题（§II，Problem 1）。** 质点模型 \(\ddot r = g + T_c/m\)，\(\dot m = -\alpha\|T_c\|\)，重力取常值；推力约束 \(0<\rho_1\le\|T_c\|\le\rho_2\)（式 3）。非凸的原因："once started, the thrusters cannot be throttled below a certain level"。另有滑翔坡约束（式 8–9，锥顶在着陆点）与 \(m(t_f)\ge m_{dry}\)；\(t_f\) 是优化变量。全文没有速度上限和推力指向约束。
2. **松弛（Problem 2，式 11–13）。** 引入松弛量 \(\Gamma\)：\(\|T_c\|\le\Gamma\)，\(\rho_1\le\Gamma\le\rho_2\)，质量方程改为 \(\dot m=-\alpha\Gamma\)，目标为 \(\min\int\Gamma\,dt\)。
3. **变量替换（式 43–49）。** \(\sigma=\Gamma/m\)，\(u=T_c/m\)，\(z=\ln m\)，得到线性动力学 \(\ddot r=u+g\)、\(\dot z=-\alpha\sigma\)，约束 \(\|u\|\le\sigma\)、\(\rho_1e^{-z}\le\sigma\le\rho_2e^{-z}\)。论文用 "a second-order-cone approximation" 处理指数项（参考点 \(z_0(t)=\ln(m_{wet}-\alpha\rho_2t)\)），近似误差上界见 [3]。
4. **离散化（§IV.D，式 59）。** 等时间间隔、在时间节点施加约束，"piecewise linear basis functions ... corresponds to first-order hold discretization"，最终是有限维 SOCP（Problem 7）。
5. **定理 1（§IV.B）。** 若非凸的最小着陆误差问题有解，算法返回的轨迹在 \([0,t_f]\) 上至多一点触及状态约束边界时，就是原问题的最优解，且在所有最优解中用燃料最少。附录更正了 Açıkmeşe & Ploen 2007 的引理 1：结论只在松弛问题最优轨迹留在可行域内部时成立；本文把它推广到"至多触边一次"。持续贴边的情形"can be constructed"但不在本文讨论范围。
6. **两步法（Algorithm 1）。** 先解松弛的最小着陆误差问题（Problem 4），再以 \(D=\|r^*(t_f)\|\) 解"指定距离内的最小燃料"问题（Problem 5）。动机：纯最小燃料问题在目标不可达时 "will simply report 'infeasible,' even though it may still be possible to land safely at some distance from the original target."
7. **飞行时间搜索（§IV.E）。** 用黄金分割搜索 \(t_f\)；下界 \(t_l\) 借 Meditch 1964，上界取 \(k_{scale}t_l\)（\(k_{scale}\) 约为 3）。\(J(t_f)\) 的单峰性只做了实验验证：100 组随机初值下平均误差 0.012%，最大 1.8%。
8. **数值结果（§V）。** JPL 自研凸优化器 G-OPT，MacBook Pro 2.4 GHz；55 个离散点。可达情形用燃料 399.4 kg、\(t_f=78.4\) s、总计算 14.3 s（含 23 次黄金分割迭代）；不可达情形最小燃料问题需 ≥414 kg 而报不可行，最小误差解落在离目标 404 m 处、用满 400 kg。全部为仿真，无飞行试验。
9. **实时性论点（摘要、§I）。** SOCP 可用内点法在多项式时间求到全局最优，"with known bounds on convergence time"，且内点法 "guaranteed to find a feasible solution if one exists"；非线性规划 "cannot provide a priori guarantees on how many iterations"。

## 关键参考文献（按本文参考文献表）

- [3] Açıkmeşe, B., Ploen, S. R., "Convex Programming Approach to Powered Descent Guidance for Mars Landing", *JGCD* 30(5), 2007, pp. 1353–1366, doi:10.2514/1.27553 — 无损凸化的原始论文（本次未取得全文）
- [6] Meditch, J. S., "On the Problem of Optimal Thrust Programming for a Lunar Soft Landing", *IEEE TAC* 9(4), 1964, pp. 477–484
- [8] Klumpp, A. R., "Apollo Lunar Descent Guidance", *Automatica* 10(2), 1974, pp. 133–146

## 对 wiki 的映射

- [无损凸化（Lossless Convexification）](../../wiki/methods/lossless-convexification.md)
- [动力下降制导（Powered Descent Guidance）](../../wiki/concepts/powered-descent-guidance.md)
- [火箭回收控制纵深路线](../../roadmap/depth-rocket-landing.md) — Stage 2
