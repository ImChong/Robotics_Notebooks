# Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time

> 来源归档（ingest）

- **标题：** Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time
- **作者：** Michael Szmuk, Behçet Açıkmeşe（University of Washington）
- **论文：** <https://arxiv.org/abs/1802.03827>（v1，2018-02-11）
- **出版物：** arXiv 注释称为 2018 AIAA Guidance, Navigation, and Control Conference（SciTech, Kissimmee）论文的修改版；arXiv 列出相关 DOI 10.2514/6.2018-0617
- **类型：** conference paper / arXiv 预印本
- **入库日期：** 2026-11-06
- **核对范围：** 用 pdftotext 抽取全文后通读 §I–V、表 1–2 与参考文献；图片只剩坐标轴噪声，式 (22) 的积分上下限版面错乱。
- **通读：** 已完成
- **一句话说明：** 把 6-DoF（质量、位置、速度、单位四元数、角速度）最短时间火箭着陆问题，用时间伸缩 + 一阶保持离散 + 虚拟控制 + 信赖域，化成一串 SOCP 子问题迭代求解，不对飞行时间做线搜索。

## 核心摘录

1. **模型（§II）。** 状态 \(x=[m, r_I, v_I, q_{B/I}, \omega_B]\in\mathbb R^{14}\)，控制为机体系推力 \(T_B\)；\(\dot m=-\alpha_{\dot m}\|T_B\|\)，四元数运动学 + 刚体欧拉方程，推力力矩 \(M_B=r_{T,B}\times T_B\)。
2. **约束（式 6–12）。** 干质量下界、滑翔坡、倾角 \(\theta\le\theta_{max}\)（用 \(1-2(q_2^2+q_3^2)\) 表示）、角速度上限、推力上下界 \(0<T_{min}\le\|T_B\|\le T_{max}\)（下界非凸）、万向节角 \(\cos\delta_{max}\|T_B\|\le e_1^\top T_B\)。终端：位置为零、给定速度与姿态、角速度为零、推力只沿机体 x 轴。
3. **自由终端时间（§III.A，式 15–16）。** 归一化时间 \(\tau\in[0,1]\)，伸缩因子 \(\sigma=(d\tau/dt)^{-1}\) 作为决策变量，\(x'(\tau)=\sigma f(x,u)\)；目标是最小化 \(\sigma\)（即飞行时间）。"our method does not perform a line search on the time-of-flight."
4. **离散化（§III.B，式 17–22）。** 在上一轮轨迹附近一阶线性化；"we assume a first-order-hold on the control over each time step"，通过状态转移矩阵积分得到 \(\bar A,\bar B,\bar C,\bar\Sigma,\bar z\)。推力下界线性化为 \(T_{min}\le \hat u^\top u/\|\hat u\|\)（式 19）。
5. **虚拟控制与信赖域（§III.C，式 23–27）。** 动力学加虚拟控制 \(\nu_k\)，用 \(w_\nu\|\bar\nu\|_1\) 惩罚，防止线性化导致的 "artificial infeasibility"；对 \(\delta x,\delta u,\delta\sigma\) 设软信赖域并加入代价。收敛判据：\(\|\bar\Delta\|_2\le10^{-3}\) 且 \(\|\bar\nu\|_1\le10^{-10}\)。
6. **数值结果（§IV）。** MATLAB + CVX + SDPT3；50 个离散点，最多 15 次迭代；所有量无量纲，"not intended to match a real-world system"。2-D 算例 10 个飞行时间初值（1–10 UT）都收敛，第 6 次迭代收敛；3-D 算例第 9 次迭代收敛。论文未给出求解耗时，实时计时留作后续工作。
7. **相关工作定位（§I）。** 无损凸化已在飞行试验中多次验证（引 G-FOLD 飞行试验与 ADAPT），但该类方法与对偶四元数方法 "were limited to fixed-final-time problems"。

## 对 wiki 的映射

- [序列凸化（Successive Convexification, SCvx）](../../wiki/methods/successive-convexification.md)
- [火箭回收控制纵深路线](../../roadmap/depth-rocket-landing.md) — Stage 3
- 开源复现：[EmbersArc/SCvx](../repos/scvx_embersarc.md)；PythonRobotics 的 `rocket_powered_landing` 示例见 [python_robotics](../repos/python_robotics.md)
