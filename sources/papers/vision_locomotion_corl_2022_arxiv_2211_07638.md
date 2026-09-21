# Legged Locomotion in Challenging Terrains using Egocentric Vision（CoRL 2022）

> 来源归档（ingest）

- **标题：** Legged Locomotion in Challenging Terrains using Egocentric Vision
- **类型：** paper / 四足 locomotion / 端到端视觉 / Teacher–Student 蒸馏
- **arXiv：** <https://arxiv.org/abs/2211.07638>（PDF：<https://arxiv.org/pdf/2211.07638.pdf>）
- **会议：** CoRL 2022 **Best Systems Paper**；PMLR v205:403–415
- **OpenReview：** <https://openreview.net/forum?id=Re3NjSwf0WF>
- **项目页：** <https://vision-locomotion.github.io/>
- **代码：** 截至 2026-09-21，**项目页未列官方 GitHub**（与 ICRA 2023 [CMS `vision_locomotion`](https://github.com/antonilo/vision_locomotion) 为不同工作）
- **作者：** Ananye Agarwal*, Ashish Kumar*, Jitendra Malik†, Deepak Pathak†（CMU + UC Berkeley）
- **入库日期：** 2026-09-21
- **一句话说明：** 在 **Isaac Gym + legged_gym** 上用 **scandots PPO → 深度 DAgger** 两阶段训练 **Unitree A1** 端到端视觉策略；**单前向 D435**、**无 metric 高程图 / 无 VIO**；实机楼梯/路缘/踏石/沟隙与户外非结构化地形 **零微调** 部署。

## 摘要级要点

- **问题：** 经典视觉 loco 依赖 **多帧深度融合高程图 + 位姿估计**，噪声大、踏石/gap 易失败，且需多相机/LiDAR；生物上人类用 **ego 视觉 + 短时记忆** 落脚，而非盯脚下高程图。
- **平台：** A1 仅 **1× 前向 RealSense**（对比 ANYmal-C 四相机+双 LiDAR、Spot 五深度）； onboard 算力有限，策略 **单向前馈 + GRU 记忆**，50 Hz 输出关节目标角。
- **Phase 1 — scandots RL（PPO + BPTT 24 步）：**
  - 观测：scandots $m_t$、本体 $x_t$、速度指令 $u^{cmd}_t$；RMA 变体另加特权 $e_t$（质心、摩擦、电机强度等）；
  - **Monolithic：** $\gamma_t=\mathrm{MLP}(m_t)$ → GRU → 关节角；
  - **RMA：** $\gamma_t=\mathrm{GRU}(m_t)$，$z_t=\mathrm{MLP}(e_t)$ → 共享 MLP base policy；
  - **无步态先验**；能量 + 碰撞/拖脚/ jerk 惩罚；6×10 地形课程（slopes / stepping stones / stairs / discrete obstacles + fractal）。
- **Phase 2 — 深度蒸馏（DAgger + BPTT）：**
  - Student 仅 **深度 $d_t$ + 本体**；ConvNet 压缩深度 → GRU 估计 $\hat{\gamma}_t$（及 RMA 的 $\hat{z}_t$）；
  - **前视相机** 须 **记忆** 后足即将落地的地形（项目页 bar stool / 后足踩空失败即此边界）；
  - Thm 2.1：Phase 2 动作贴近 Phase 1 时 return 接近最优（Lipschitz 假设）。
- **仿真结果（Table 1，单策略全地形）：** Monolithic / RMA 总 mean time to fall **~275 s** vs blind **~175 s**、noisy elevation **~148 s**；踏石 forward displacement **~19–21 m** vs baseline **~1 m**。
- **实机（Fig.4）：** upstairs **100%**（13 级）vs blind **0%**；downstairs blind **100%** 但 **高冲击摔落步态**（损硬件）；踏石 **94%**；gap **100%**（26 cm）；urban **24 cm 楼梯 / 26 cm 路缘**；户外泥阶/河滩。
- **局限（§6）：** sim–real 视觉/地形失配需 **回仿真重训**；无前视外的几何（高 dip 路缘、后足记忆误差）；未开源代码。

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/paper-vision-locomotion-egocentric.md`](../../wiki/entities/paper-vision-locomotion-egocentric.md)
- 交叉更新：[`wiki/tasks/stair-obstacle-perceptive-locomotion.md`](../../wiki/tasks/stair-obstacle-perceptive-locomotion.md)、[`wiki/entities/extreme-parkour.md`](../../wiki/entities/extreme-parkour.md)、[`wiki/entities/paper-rma-rapid-motor-adaptation.md`](../../wiki/entities/paper-rma-rapid-motor-adaptation.md)、[`wiki/entities/legged-gym.md`](../../wiki/entities/legged-gym.md)
