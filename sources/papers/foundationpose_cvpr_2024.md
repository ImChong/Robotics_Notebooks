# FoundationPose：新物体的统一 6D 位姿估计与追踪（CVPR 2024）

- **英文题目：** FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects
- **作者：** Bowen Wen、Wei Yang、Jan Kautz、Stan Birchfield
- **出处：** CVPR 2024，17868–17879
- **CVF 官方论文：** https://openaccess.thecvf.com/content/CVPR2024/html/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.html
- **PDF：** https://openaccess.thecvf.com/content/CVPR2024/papers/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.pdf
- **官方项目：** https://nvlabs.github.io/FoundationPose/
- **官方代码：** https://github.com/NVlabs/FoundationPose
- **归档日期：** 2026-11-12
- **核对范围：** CVF 原文摘要、引言、方法框架图、神经物体场、初始化/姿态细化、pose ranking 及部分公式；后续实验与附录未逐页核对。
- **通读：** 未完成

## 核心摘录

- 同时涵盖 model-based（给定 CAD）与 model-free（给定少量参考 RGB-D 图像，先建立神经物体表征）两种新物体设置。
- 实现包括 6D pose hypotheses、可微/可渲染物体表征、render-and-compare pose refiner、pose scorer，以及跟踪能力。
- 对 4D-HOF：提供物体姿态初始化。4D-HOF 在论文附录中进一步说明了 **逐帧配准 + 相邻旋转测地距阈值 60° + 遮挡区域排除的 mask IoU 过滤**，并非原样启用 FoundationPose 默认 tracking。

## 对 wiki 的映射

- [4D-HOF 实体](../../wiki/entities/paper-4d-hof.md) — 当前按方法依赖记录，尚未创建独立 FoundationPose 实体。
