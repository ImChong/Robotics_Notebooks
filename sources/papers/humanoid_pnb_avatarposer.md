# AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/AvatarPoser__Articulated_Full-Body_Pose_Tracking_from_Sparse_Motion_Sensing/AvatarPoser__Articulated_Full-Body_Pose_Tracking_from_Sparse_Motion_Sensing.html>
- **分类：** 14_Human_Motion
- **arXiv：** <https://arxiv.org/abs/2207.13784>
- **入库日期：** 2026-07-10
- **一句话说明：** 在 VR/AR 头显平台上驱动关节化化身，通常只能拿到稀疏运动信号——头显（HMD）+ 两个手柄的三点位姿。AvatarPoser 提出从这些稀疏运动感知实时重建全身关节姿态：基于 Transformer 的网络从稀疏输入预测全身关节旋转，并把全局运动与局部姿态解耦（用稳定的全局参考），再用逆运动学（IK）微调手臂，使预测末端精确对齐输入关节。AvatarPoser 在大规模 AMASS 动作数据上达 SOTA，为头显平台的化身驱动奠定基础（ECCV 2022）。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-avatarposer-articulated-full-body-pose-tracking](../../wiki/entities/paper-notebook-avatarposer-articulated-full-body-pose-tracking.md).

## 对 wiki 的映射

- [paper-notebook-avatarposer-articulated-full-body-pose-tracking](../../wiki/entities/paper-notebook-avatarposer-articulated-full-body-pose-tracking.md)
- 分类父节点：[paper-notebook-category-14-human-motion](../../wiki/overview/paper-notebook-category-14-human-motion.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/AvatarPoser__Articulated_Full-Body_Pose_Tracking_from_Sparse_Motion_Sensing/AvatarPoser__Articulated_Full-Body_Pose_Tracking_from_Sparse_Motion_Sensing.html>
- 论文：<https://arxiv.org/abs/2207.13784>
