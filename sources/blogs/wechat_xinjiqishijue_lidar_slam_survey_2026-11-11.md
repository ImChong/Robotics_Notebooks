# 激光雷达SLAM算法最新综述，一文览尽所有主流框架

> 来源归档（微信公众号转载的综述）

- **标题：** 激光雷达SLAM算法最新综述，一文览尽所有主流框架
- **类型：** blog（综述转载）
- **作者：** 刘铭哲 等（陆军工程大学通信工程学院）；转载公众号「新机器视觉」
- **原始链接：** https://mp.weixin.qq.com/s/d7sJHfQXK3nbe3p1xKp22Q
- **发表日期：** 2026-11-11（转载日期）
- **入库日期：** 2026-11-11
- **一句话说明：** 对激光雷达 SLAM 的框架综述：前端里程计与配准、后端（滤波 / 图优化）、回环、建图；2D / 3D 激光里程计、多传感器融合、深度学习 SLAM、数据集与指标、展望。
- **核对范围：** 文字全文已读；文中表 1–5 在抓取文本中缺失，未核对；评论区未读。文章没有给出所依据的原始论文出处，我没有找到原论文，所以无法核对综述与原文是否一致。文中各算法的说法已分别对照各算法的论文（或仓库）核对，结果写在各节点与概念页里。
- **通读：** 已完成
- **说明：** 综述本身不含新实验，只汇总各算法；其中的数字与结论是作者表述，以各节点的核对为准。

## 文章结构与要点

- **SLAM 框架**：前端里程计（去畸变、配准：ICP 及 PP / PL / NICP / IMLS / GICP / VGICP、NDT、学习型 FCGF / SpinNet / Lepard / REGTR）；后端（滤波：KF / EKF / UKF / ESKF / PF；图优化：Ceres / g2o / iSAM / GTSAM / SE-Sync）；回环（Scan Context、Scan Context++、FPFH）；建图（栅格、点云、OctoMap、SLAMesh、语义）。
- **2D SLAM**：FastSLAM、GMapping、Karto、Hector、Cartographer。
- **3D 激光里程计**：LOAM、A-LOAM、LeGO-LOAM、loam_livox、DLO（NanoGICP）、CT-ICP（文中写作「CP-ICP」）、基于强度的 SLAM。
- **多传感器融合**：LIO-SAM、LVI-SAM、R2LIVE、R3LIVE、R3LIVE++、FAST-LIO、FAST-LIO2、Faster-LIO、FAST-LIVO、Point-LIO。
- **深度学习 SLAM**：OverlapNet、OverlapTransformer、SuMa++、SA-LOAM、ASL-SLAM、自监督里程计、EfficientLO-Net。
- **数据集与指标**：KITTI、SemanticKITTI、ApolloScape、NTU-VIRAL 等；ATE / RPE；evo。
- **展望**：深度学习、多传感器融合、多智能体 SLAM（DOOR-SLAM、Kimera-Multi、D2SLAM、Swarm-LIO）、鲁棒性与动态场景。

## 算法 → 节点

见概念页 [lidar-slam-survey-2026](../../wiki/concepts/lidar-slam-survey-2026.md) 的映射表（新建 / 复用 / 未建节点逐项标明）。
