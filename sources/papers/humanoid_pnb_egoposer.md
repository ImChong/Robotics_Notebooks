# EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/EgoPoser__Robust_Real-Time_Egocentric_Pose_Estimation_from_Sparse_Sensors/EgoPoser__Robust_Real-Time_Egocentric_Pose_Estimation_from_Sparse_Sensors.html>
- **分类：** 14_Human_Motion
- **arXiv：** <https://arxiv.org/abs/2308.06493>
- **入库日期：** 2026-07-10
- **一句话说明：** 仅用头与手位姿做全身第一视角姿态估计是头显平台驱动化身的活跃方向，但现有方法过度依赖室内动捕空间（数据录制环境），且假设关节连续跟踪与统一体型。EgoPoser 在更真实的设定下做到鲁棒：手部位置/朝向只有进入头显视野（FoV）时才被跟踪——即稀疏且间歇的观测。它还有三点关键贡献：① 在间歇手部跟踪下仍鲁棒建模全身姿态；② 全局运动分解——独立于全局位置预测全身姿态；③ 高效的 SlowFast 模块设计，在捕捉更长运动时序的同时保持算力高效；并能跨不同用户体型泛化。ECCV 2024。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-egoposer-robust-real-time-egocentric-pose-estima](../../wiki/entities/paper-notebook-egoposer-robust-real-time-egocentric-pose-estima.md).

## 对 wiki 的映射

- [paper-notebook-egoposer-robust-real-time-egocentric-pose-estima](../../wiki/entities/paper-notebook-egoposer-robust-real-time-egocentric-pose-estima.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/EgoPoser__Robust_Real-Time_Egocentric_Pose_Estimation_from_Sparse_Sensors/EgoPoser__Robust_Real-Time_Egocentric_Pose_Estimation_from_Sparse_Sensors.html>
- 论文：<https://arxiv.org/abs/2308.06493>
