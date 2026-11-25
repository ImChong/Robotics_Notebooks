# LynxScenes: A Large-Scale Real-World Benchmark for Outdoor Wheel-Legged Robot Perception, Navigation, Imagination and Reconstruction（arXiv:2610.18093）

> 来源归档（paper / arXiv v1）

- **标题：** LynxScenes: A Large-Scale Real-World Benchmark for Outdoor Wheel-Legged Robot Perception, Navigation, Imagination and Reconstruction
- **arXiv：** <https://arxiv.org/abs/2610.18093>（v1，2026-10-13；PDF：<https://arxiv.org/pdf/2610.18093>）
- **全文：** <https://arxiv.org/html/2610.18093>
- **项目页：** <https://developer.deeprobotics.cn/research/articles/lynx-scenes/index.html>
- **作者：** Jun Tian, Hao Wu, Ruo Chen, Yiyi Liao, Hongyi Lin, Jincheng Yu, Chao Li*, Yingjie Niu*†
- **机构：** 云深处科技（DEEP Robotics）；伊利诺伊大学厄巴纳-香槟分校（UIUC）；香港科技大学（HKUST）；浙江大学；清华大学；麻省理工学院（MIT）
- **入库日期：** 2026-11-21
- **核对范围：** 通读 `https://arxiv.org/html/2610.18093`（v1，2026-10-13）全文主干：Abstract、§1–§2、§3 数据集（3.1 平台传感、3.2 采集同步、3.3 轨迹构造、3.4 3D 物体/道路结构/三层 20 条导航指令标注、3.5 统计与 8:1:1 划分）、§4 四大基准任务与 QNS 指标、§5 实验结果（5.1–5.4 轨迹预测、BEV 感知、未来场景预测、多视角重建，Table 2–5）、§6 结论。边界：数据集与评测工具标注 SOON（ICLR 2027 审稿中），未在本地运行基线评测。
- **通读：** 已完成
- **一句话说明：** 基于实体 DEEP Robotics LYNX M20 轮足机器人采集的室外大规模真实多模态基准，包含约 100K 个 30 秒片段、3.0M 标注帧、833.6 小时和 3,294.7 km 里程，在同一实体与场景分布下统一评测指令条件轨迹预测（含伪闭环 QNS）、BEV 感知、未来场景想象与高斯泼溅重建四类任务。

## 核心数据与评测要点（原文核对）

- **平台与传感（§3.1–3.3, Appendix B.2）：** DEEP Robotics **LYNX M20** 轮足四足机器人，配备 4 路环视鱼眼相机（1920×1536 @ 10 Hz）、2 台 96 线 RoboSense Airy LiDAR（360°×90°、60 m @ 10 Hz）、IMU、6-DoF LIO 与机器人状态流 $(v_x, v_y, \omega_z)$，统一表达于机器人中心 FLU 坐标系。以 1 Hz 采样 3.0M 标注帧，未来参考轨迹由状态流积分并重采样至 10 Hz。
- **标注体系（§3.4–3.5）：**
  - **3D 物体：** **20.82M** 个带朝向 3D 框（平均每帧 6.94 个，76.52% 位于 20 m 近距内），覆盖 27 类与持久化实例 ID、动态状态和速度（行人 35.98%、滑板车 14.12%、汽车 13.46% 等）。
  - **道路结构：** **13.89M** 条标注（平均每帧 4.63 条），覆盖 Road Curb、Sidewalk Curb、Road–Sidewalk Dividing Line（折线）与 Crosswalk（闭合轮廓）。
  - **三层 20 条行为导航指令：** Primary navigation（6 类，100% 覆盖）、Special motion（9 类，35.91% 覆盖）、Obstacle interaction（5 类，22.42% 覆盖）；层内互斥、跨层可并行。
  - **划分：** 按 30 s 片段以 **8:1:1** 划分 train / val / test，同一连续采集 session 归入同一 split 以防时间/场景泄漏。
- **四大基准任务与代表结果（§4–§5）：**
  - **任务 1 指令条件轨迹预测：** 输入过去 4 个时刻 16 张鱼眼图 + 当前运动状态 + 三层指令，预测未来 1 s（$10 \times 3$）平面位姿。10 个 VLN/VLA 基线中，**NaVILA** 在 QNS 取得最高 **0.843**（StreamVLN 0.833、π₀.₅ 0.819、OpenVLA 0.802）；论文指出单纯模仿误差最低未必安全或遵从指令。
  - **任务 2 BEV 感知（Table 3）：** **SparseDrive**（Cam.+Pose）取得最高道路结构 Line mAP **21.08**；**BEVFusion**（Cam.+LiDAR）取得最高 3D 目标 Box mAP **67.11** 与 HDS **64.48**。
  - **任务 3 未来场景预测（200 片段子集，Table 4）：** **WorldStereo** 取得最佳 FVD（386.2），**LingBot-World** 领跑 Q-Align-I/V 与 LAION-Aes，无单一模型全胜。
  - **任务 4 多视角重建（Table 5）：** **ReCoSplat** 取得最佳 PSNR（15.94）、LPIPS 与多尺度结构相似度。

## 对 wiki 的映射

- [LynxScenes（arXiv:2610.18093）](../../wiki/entities/paper-lynxscenes.md)
