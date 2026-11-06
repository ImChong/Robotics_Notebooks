# Convex Optimization for Trajectory Generation（Malyuta 等教程）

> 来源归档（ingest）

- **标题：** Convex Optimization for Trajectory Generation: A Tutorial on Generating Dynamically Feasible Trajectories Reliably and Efficiently
- **作者：** Danylo Malyuta, Taylor P. Reynolds, Michael Szmuk, Thomas Lew, Riccardo Bonalli, Marco Pavone, Behçet Açıkmeşe（University of Washington、Stanford）
- **论文：** <https://arxiv.org/abs/2106.09125>（v1，2021-06-16，68 页）
- **出版物：** arXiv 版只印 "Preprint"。正式版按 [SCPToolbox.jl README](../repos/scptoolbox_jl.md) 的引用为 *IEEE Control Systems Magazine* 42(5):40–113, 2022, DOI 10.1109/MCS.2022.3187542（未在期刊页面复核）
- **代码：** 文中 Fig. 2 指向 `github.com/dmalyuta/scp_traj_opt` 的 `csm` 分支，该仓库现更名为 [SCPToolbox.jl](../repos/scptoolbox_jl.md)
- **类型：** tutorial / survey
- **入库日期：** 2026-11-06
- **核对范围：** 用 pdftotext 抽取全文后读了前言、引言、Part I（LCvx 定理与火箭着陆问题）、Part II（SCvx 与 GuSTO 的算法、定理与讨论）、各侧栏、Part III 火箭着陆算例、结论与参考文献；Part III 的四旋翼与自由飞行机器人算例只略读。
- **通读：** 未完成
- **一句话说明：** 系统讲解无损凸化（LCvx）与两种序列凸规划（SCvx、GuSTO）的轨迹生成教程，附 Julia 实现与火箭着陆、四旋翼、自由飞行机器人三个算例。

## 核心摘录

1. **LCvx 思想（Part I）。** "Pontryagin's maximum principle is used to show that a convex relaxation of a nonconvex problem finds the globally optimal solution to the original problem." 核心是证明松弛不等式 \(g_1(u)\le\sigma\) 在最优解处取等号。定理 1 的条件：完全能控（Condition 1）+ 横截性相关的线性无关条件（Condition 2）。
2. **LCvx 的边界。** "LCvx tends to apply only to very specific problems"；一般凸状态约束只在约束 "active pointwise in time" 时有保证；动力学非线性时松弛问题仍非线性。3-DoF 火箭模型 "accurate as long as attitude can be controlled in an inner loop faster than the outer translation control loop."
3. **SCvx vs GuSTO（Part II）。** SCvx 在离散化问题上给出 KKT 意义的保证，信赖域是硬约束，用虚拟控制处理人为不可行；GuSTO 在连续时间问题上给出 Pontryagin 意义的保证，信赖域和非凸路径约束都用逐步加大的软惩罚，但要求代价对控制二次、动力学控制仿射。SCvx 超线性收敛；虚拟控制收敛到零时得到驻点。
4. **飞行背景（引言、Part II）。** JPL 在 Masten Xombie 上演示了 Mars 精确着陆凸优化制导（引 [37–39]）；"We hypothesize that LCvx is also a close relative of the SpaceX Falcon 9 terminal landing algorithm"（作者推测）；"Although SpaceX's methods are undisclosed, we know that convex optimization is used by the Falcon 9 rocket"（依据 Blackmore 2016）；NASA SPLICE 项目中 SCvx 作为实验载荷在 Blue Origin New Shepard 上测试（引 NASA 2020 页面）。
5. **火箭着陆算例（Part III）。** 3-DoF LCvx：\(g=-3.71\) m/s²，干/湿质量 1505/1905 kg，Isp 225 s，推力 4.971–13.258 kN，滑翔坡 86°，指向角 40°，速度上限 500 km/h；零阶保持、Δt = 1 s，黄金分割搜索得 \(t_f^*=75\) s；滑翔坡只在两处激活，约 40–70 s 有一段最小推力弧。该算例未报求解器与耗时。
6. **实践建议（Part II/III）。** 初值用直线插值（火箭推力从 \(-m_{wet}g\) 插到 \(-m_{dry}g\)，四元数用 SLERP）；一阶保持（FOH）离散在最优性、可行性、计算量间折中好，但只在节点施加约束会出现节点间"clipping"；"SCvx and GuSTO are not scale-invariant"，变量要缩放到约 [0,1]，"Without variable scaling, SCP algorithms usually have a very hard time converging"；虚拟控制罚权从 \(10^2\)–\(10^4\) 试起，收敛后虚拟控制非零时先加大罚权。

## 对 wiki 的映射

- [无损凸化（Lossless Convexification）](../../wiki/methods/lossless-convexification.md)
- [序列凸化（Successive Convexification, SCvx）](../../wiki/methods/successive-convexification.md)
- [火箭回收控制纵深路线](../../roadmap/depth-rocket-landing.md) — Stage 2 / 3
