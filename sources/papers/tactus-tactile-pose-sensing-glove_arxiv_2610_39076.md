# Tactus（arXiv:2610.39076）

> 来源归档（paper）

- **标题：** Tactus: Tactile & Pose Sensing Glove for Natural Dexterity Data Collection
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.39076>
- **PDF：** <https://arxiv.org/pdf/2610.39076>
- **作者：** Guan, Li, Granberry, Rachael, Aseeri, Sahar, Rostami, Sahar, Niehues, Taylor, Marri, Sai Dhanooshi, Huang, Mia, Zhang, Dingtian et al.
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** Tactus：用于自然灵巧操作数据采集的触觉与姿态感知手套。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 人类灵巧操作示范需要准确的手部姿态和接触力信息，但相机在手物遮挡时难以恢复姿态，传统传感手套又可能影响自然抓取动作。
3. **周报所述创新点：** 提出织物集成式传感手套，结合电容触觉阵列与 IMU 姿态跟踪。系统包含 47 个触觉传感位置和 16 个六轴 IMU，并提供跨设备标定流程。在操作过程中，指尖关键点相对误差约 11.0 mm，可用于接触感知机器人策略的数据采集。
4. **arXiv 摘要（英文节选）：** Collecting naturalistic human manipulation demonstrations with accurate contact force and hand pose is essential for robot dexterous manipulation policy and foundational world model learning - yet no existing wearable platform provides both calibrated fingertip force sensing and occlusion-resilient hand tracking in a form factor that preserves natural manipulation behavior. We present Tactus, a textile-integrated sensing glove that co-optimizes three design pillars: (1) a thin, conformal 3D-knit form factor (1.5 mm fingertip stackup) designed to minimize interference with natural manipulation, (2) strategically placed capacitive tactile sensors at manipulation-relevant regions informed by empirical contact studies, and (3) IMU-fused pose tracking that maintains accuracy through the hand-object occlusions where manipulation data is most valuable. Tactus includes a documented calibration pipeline for cross-glove consistency and per-session calibration for subject-dependent correction.

**对 wiki 的映射**

- [paper-tactus-tactile-pose-sensing-glove](../../wiki/entities/paper-tactus-tactile-pose-sensing-glove.md)
