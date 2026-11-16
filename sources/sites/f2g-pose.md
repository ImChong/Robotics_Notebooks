# F2G-Pose 官方项目主页核验

- **类型：** site / project page
- **项目页：** https://jrryzh.github.io/F2G-Pose/
- **入库日期：** 2026-11-09
- **标题：** F2G-Pose — Geometry-Aware Foundation Feature Lifting for Direct RGB-D Category-Level Object Pose Estimation
- **论文：** https://arxiv.org/abs/2610.31623
- **代码：** https://github.com/jrryzh/F2G-Pose
- **模型权重：** https://huggingface.co/J3rr1/F2G-Pose
- **核对范围：** 读取官方项目页全部文字，包括 Method、Film、Examples、Results、Read & Build；核对页面链接可分别进入论文、代码、权重。视频列举为项目页所标示案例，未进行逐帧独立测量。
- **通读：** 已完成

## 开源判断

截至 2026-11-09，**代码与权重已公开**，非仅承诺将来发布；提供 Gradio demo、CLI、Python API、训练/评测说明。F2G-Pose 集成部分 Apache-2.0；RADIO、Omni6DPose 数据、SAM、PointNet++ 等第三方依赖按各自许可；RADIO 权重需要单独下载。

## 项目页证据

- 一帧 segmented RGB-D → 6D pose、metric size、camera-frame shape（single pass）。
- RADIO 冻结图像特征抬升到点云 → DGCNN + MoE Transformer → pose-size 与 shape 两分支。
- 页面列出 ROPE IoU50 AUC 27.8 vs GenPose++ 19.1；SOPE 43.7 vs 31.9；28 FPS pose-only、23.7 FPS full；ARX 放置 68/80。
- 展示棋子、笛子、橡皮、透明酒杯/反光器皿、机器人工作空间等定性案例；不将单张可视化视作统计鲁棒性证明。

## 对 wiki 的映射

- [F2G-Pose 实体](../../wiki/entities/paper-f2g-pose.md)
- [论文深读归档](../papers/f2g_pose_arxiv_2610_31623.md)
- [源码复核归档](../repos/jrryzh_f2g_pose.md)
