# EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere

> 来源归档（ingest · Robot Learning Paper Notebooks 深读笔记 + arXiv v3 全文通读）

- **标题：** EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/EgoPoser__Robust_Real-Time_Egocentric_Pose_Estimation_from_Sparse_Sensors/EgoPoser__Robust_Real-Time_Egocentric_Pose_Estimation_from_Sparse_Sensors.html>
- **分类：** 14_Human_Motion
- **arXiv：** <https://arxiv.org/abs/2308.06493>（v3，2024-09-06，ECCV 2024，18 页）
- **项目主页：** <https://siplab.org/projects/EgoPoser>
- **官方代码：** <https://github.com/eth-siplab/EgoPoser>
- **核对范围：** 全文通读论文文本 §1–§5、Table 1–5（HPS 外推、体型感知、FoV 视野遮挡、全局运动分解与 SlowFast 消融）、Fig. 1–11（Quest 2 实机定性与位置偏移鲁棒性）及公式定义。
- **通读：** 已完成
- **入库日期：** 2026-07-10（原文通读复核：2026-11-14）
- **一句话说明：** 仅用头与手位姿做全身第一视角姿态估计时，EgoPoser 通过真实 FoV 间歇遮蔽建模、时间/仅水平空间全局运动分解、SlowFast 双分支特征融合以及无标定隐式体型联合优化，在跨场景（HPS）、手出视野和跨体型条件下实现 >600 FPS 实时全身姿态重建（ECCV 2024）。

## 核心摘录（策展与核对摘要）

- **间歇观测建模**：按头显水平/垂直视角（180°、120°、90°）将视野外手部特征置零，Table 3 下明显优于全可见训练模型与随机丢帧基线 FLAG。
- **全局运动分解（GMD）**：时间归一化减去首帧平移 + 空间归一化仅将水平坐标对齐头部（保留竖直坐标），Table 1 在 HPS 大场景外推 5 段序列取得最低 MPJPE/MPJVE。
- **SlowFast 与体型联合估计**：80 帧窗口压成 40 帧（0.33G FLOPs，4.12M 参数）；SMPL-H 前向运动学误差隐式监督体型参数 $\beta \in \mathbb{R}^{16}$ 并辅以 L1 正则。

## 对 wiki 的映射

- [paper-notebook-egoposer-robust-real-time-egocentric-pose-estima](../../wiki/entities/paper-notebook-egoposer-robust-real-time-egocentric-pose-estima.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/EgoPoser__Robust_Real-Time_Egocentric_Pose_Estimation_from_Sparse_Sensors/EgoPoser__Robust_Real-Time_Egocentric_Pose_Estimation_from_Sparse_Sensors.html>
- 论文：<https://arxiv.org/abs/2308.06493>
