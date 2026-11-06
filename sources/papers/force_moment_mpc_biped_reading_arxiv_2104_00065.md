# Force-and-moment-based MPC 双足论文通读（凸 MPC 视角）

> 来源归档（原文通读 · 支撑 `wiki/methods/model-predictive-control.md`）

- **标题：** Force-and-moment-based Model Predictive Control for Achieving Highly Dynamic Locomotion on Bipedal Robots
- **作者：** Junheng Li, Quan Nguyen（USC）
- **类型：** paper / mpc / legged-locomotion
- **原文：** <https://arxiv.org/abs/2104.00065>（2026-11 通过 arXiv PDF v2 获取全文）
- **一句话说明：** 把四足凸 MPC（Di Carlo et al.）推广到 10 自由度双足：以足端三维力 + 二维力矩为输入，在简化刚体动力学上建线性 MPC，转成 QP 实时求解，再用 J^T 映射成关节力矩。
- **核对范围：** 通读全文 §I–§V（含 §II 机器人与仿真、§III 简化动力学 / MPC / QP / 摆动腿、§IV 全部仿真实验、§V 结论）；参考文献仅浏览条目，未逐篇阅读（含被其扩展的 Di Carlo 2018 四足凸 MPC，该文未在此归档内读）。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **控制输入（§III-A，式 1–4）：** u = [F1, F2, M1, M2]，每足 3D 力 + 绕 y、z 的 2D 力矩（不含绕 x 力矩）；质心平移与 Ḣ≈I_G·ω̇ 对 u 线性。
- **三种简化模型对比（§III-A、§IV-A）：** 模型 1（每足 2 接触点、3D 力，类四足）无法做俯仰；模型 2（每足单点、3D 力 + 3D 力矩）无法做横滚；模型 3（单点、3D 力 + 2D 力矩）可完成全部三维姿态运动，被采用。
- **线性化（式 5–12）：** 小 roll/pitch 假设下 Θ̇≈Rz(ψ)ω；状态 x=[Θ, pc, ω, ṗc, g]（重力并入状态）得 ẋ=Â_c(ψ)x+B̂_c u；Â 用参考轨迹平均偏航，B̂ 用期望足位（首步用当前状态）。
- **MPC 目标与约束（式 13–19）：** 最小化 Σ(x−x_ref)ᵀQ(x−x_ref)+‖u‖_R；动力学等式；摩擦锥（四棱锥近似）|Fx|,|Fy|≤μFz；Fmin≤Fz≤Fmax（下界为正以保持接触）；|τ|≤τmax；摆动足输入强制为 0（等式约束）。
- **QP（式 20–25）：** X=A_qp x0+B_qp U，代入得 min ½UᵀhU+Uᵀf，s.t. CU≤d，A_eq U=b_eq；结论称 QP 可在 <1 ms 内求解。
- **映射与摆动腿（式 26–31）：** 支撑腿 τ=Jᵀu；摆动腿用笛卡尔 PD 虚拟弹簧阻尼，落足点 p_foot_d=p_hip+ṗc·Δt/2（沿用四足策略）；步态纯时间触发，周期 0.3 s。
- **仿真设置（§IV）：** MATLAB/Simulink + Spatial v2；MPC 采样 0.03 s、仿真 1 kHz；一个步态周期 10 个预测步；权重 Qx=Qy=50, Qz=100, Qφ=Qθ=100, Qψ=20，其余 1。机器人 11.84 kg，电机沿用 Unitree A1（峰值 33.5 Nm）。
- **结果：** 变速前进最大速度偏差 0.076 m/s；粗糙地形（台阶最高 0.075 m、相邻高差 0.055 m）上 1.6 m/s 行走，关节力矩未饱和；同一套参数完成行走、跳跃（双支撑 + 每周期最后 1/4 腾空）与跑步。
- **局限（§V）：** 仅物理仿真，无硬件验证；作者计划优化公式以支持更激进运动并做实机实验。

## 对 wiki 的映射

- [`wiki/methods/model-predictive-control.md`](../../wiki/methods/model-predictive-control.md)
