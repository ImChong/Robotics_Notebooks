# DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination（arXiv:2301.10602）

- **通读：** 已完成
- **核对范围：** 2026-11-20 通读 `arXiv:2301.10602v2`（3 Mar 2023）论文全文（pdftotext 506 行，含 §II 方法、§III 实验、结论与参考文献；论文无附录），核实：actor 输入为 H=5 帧本体历史 + CENet 输出的体速 `v_t` 与隐变量 `z_t`；critic 额外看扰动力 `d_t` 与高度图 `h_t`；12 维动作为站立姿态偏移 + PD（Kp=28、Kd=0.7，策略/CENet 50 Hz，PD 200 Hz）；CENet 为共享编码器 + 速度头（MSE）+ β-VAE 重建下一帧 `o_{t+1}`；AdaBoot `p_boot = 1 − tanh(CV(R))`；Isaac Gym 4096 并行环境训练 1000 iter（约 1 小时 ≈ 现实 46 天）；PPO clip 0.2 / GAE 0.95 / γ 0.99 / lr 1e-3；推扰鲁棒性（最大推速 1.121±0.164 m/s、存活率 95.23±1.61%，对照 Baseline 0.511 / 20.51%）；Unitree A1 室外 430 m（A）与 465 m（B，爬升 22 m）长距离行走；局限为需先接触障碍、高台阶需引入外感知。论文未给出官方源码仓库链接（仅项目页），故未读源码。
- **标题：** DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination
- **会议：** ICRA 2023
- **链接：** <https://arxiv.org/abs/2301.10602>
- **扩展：** [DreamWaQ++（arXiv:2409.19709）](dreamwaq_plus_arxiv_2409_19709.md)（T-RO 2026，多模态点云）
- **一句话说明：** 四足**盲走**单阶段 RL：CENet 从本体历史估计隐式地形上下文与体速，非对称 Actor–Critic 在仅本体部署下实现鲁棒 locomotion，为后续障碍感知扩展奠基。

## 核心机制

- **CENet（Context Estimation Network）：** 编码本体历史 → 隐式地形想象 + 速度估计。
- **训练：** 特权 critic + 部分观测 actor；无需显式高度图或外感知传感器。
- **局限：** 楼梯/缺口等需**接触后**才能调整步态；前瞻障碍见 DreamWaQ++ 与 PIE 路线。

## 对 wiki 的映射

- [DreamWaQ 方法页](../../wiki/methods/dreamwaq.md)
- [DreamWaQ++ 实体页](../../wiki/entities/dreamwaq-plus.md)
- [Privileged Training](../../wiki/concepts/privileged-training.md)
- 飞书 Know-How「DreamWaq盲走一阶段鲁棒行走」：[humanoid-motion-control-know-how-technology-map](../../wiki/overview/humanoid-motion-control-know-how-technology-map.md)
