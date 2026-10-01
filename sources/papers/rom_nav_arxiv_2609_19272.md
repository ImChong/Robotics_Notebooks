# Learning Safe Humanoid Navigation from Reduced Order Models（arXiv:2609.19272）

> 来源归档（paper）

- **标题：** Learning Safe Humanoid Navigation from Reduced Order Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.19272>
- **PDF：** <https://arxiv.org/pdf/2609.19272>
- **项目页：** <https://wdc3iii.github.io/rom-nav/>
- **作者：** William D. Compton, Zachary Olkin, Ryan Bena, Aaron D. Ames
- **机构：** California Institute of Technology（CDS）；Amazon Safe Autonomy Frontiers (SAF) Lab
- **投稿：** ICRA 2027 under review（8 pages, 6 figures；arXiv v1 2026-09-16）
- **入库日期：** 2026-09-18（骨架）；**2026-10-01**（深读 ingest）
- **一句话说明：** **RoM-Nav** 先在单积分器降阶动力学 + 全 3D LiDAR 上训导航，再用 KL+PPO kickstart 全人形（冻结 locomotion）；部署层叠 **Poisson 方程 CBF** 滤除 OOD 碰撞且不掉成功率；G1 无地图跨楼层 **>10 m** 高差、**100 m** 路径。

## 开源状态

- **待发布**（步骤 2.5，**2026-10-01**）：[项目页](../sites/rom-nav.md) 无 Code 链接，BibTeX `eprint` 占位 `TODO`。

## 核心摘录

1. **问题分解：** 单阶段 RL 直接在全身人形上训导航难以扩展到多层/多 story 地形（楼梯等复杂接触）；本文拆为 **RoM 导航策略**（occupancy 网格上的单积分器+航向，撞格时投影滑沿边界）→ **kickstart 全阶人形策略**（同一 LiDAR/深度观测，**冻结 locomotion** 闭环），损失为 \(\mathcal{L}=\mathcal{L}_{\mathrm{PPO}}+\lambda\mathcal{L}_{\mathrm{KL}}(\pi,\pi^R)\)，\(\lambda\) 前 100 iter 为 1，1100 iter 衰减至 0.05 并保持至 2000 iter。

   **对 wiki 的映射：** [paper-rom-nav](../../wiki/entities/paper-rom-nav.md)「方法 · kickstart」；与 [Generate, Track, Improve](../../wiki/entities/paper-generate-track-improve.md) 同属 Caltech 系 **冻结下层 locomotion + 上层规划/导航** 分工。

2. **感知与课程：** LiDAR 距离图 + 下视深度经 **去噪 VAE 预训练 CNN**（深度/XYZ/掩码重建 + 地面/障碍/楼梯/坡道分割），2M 图像（均匀 + 楼梯坡道过采样）训 10 epoch 后 **冻结**；无预训练 SR 骤降。程序化 **multi-story tile** + CUDA 测地线目标（≤30 m）+ **30% 楼梯/坡道口 spawn**（对齐 locomotion gait library 相位）使 cross-level 目标可学。

   **对 wiki 的映射：** [paper-rom-nav](../../wiki/entities/paper-rom-nav.md)「工程实践 · 编码器与采样」；对比 [navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md) 中「全局地图」路线，本文策略 **严格无地图**（GLIM 仅硬件对齐初值）。

3. **仿真结果（1024 初值，Table II）：** RoM-Nav SR@45 **82.3%** / SR@120 **92.8%**，接近 RoM 上界（cyl 84.1/93.6%），较 Single-Stage（62.2/81.7%）与 RoM zero-shot 到人形（74.2/88.1%）明显提升；**cross-level** 对比 RoM-Nav − Single-Stage @45s **+34.6 pp**（95% CI 显著），主要降 **fall rate**（120s cross-level：16.3%→8.3%）。

   **对 wiki 的映射：** [paper-rom-nav](../../wiki/entities/paper-rom-nav.md)「实验与评测」。

4. **Poisson 安全滤波：** 点云栅格 0.05 m、去地面、按机体半径膨胀，解 **Poisson 方程** 得 CBF \(h\)，QP 投影速度命令 \(v_{\mathrm{safe}}\)（\(\alpha=0.75\)）；OOD/对抗障碍下无 CBF 碰撞 2/10、4/10，有 CBF **0/10** 且成功率仍 10/10（时间略增）。真机长程：**10 m** 垂直爬升（超训练 8 m 极值）、**100 m** 路径（超 30 m 训练 cap），**无碰撞**；透明玻璃需人工挡板（LiDAR 局限）。

   **对 wiki 的映射：** [paper-rom-nav](../../wiki/entities/paper-rom-nav.md)「安全层」；概念对照 [CLF vs CBF](../../wiki/comparisons/clf-vs-cbf.md)（本文 CBF 来自 occupancy Poisson，非学习值函数）。

## 参考链接

- [具身智能小站 10 篇盘点（策展入口）](../blogs/wechat_embodied_station_10_papers_contact_wm_2026-09-18.md)
