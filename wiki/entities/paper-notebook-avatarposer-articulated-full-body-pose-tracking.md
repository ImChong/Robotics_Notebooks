---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2207.13784"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_avatarposer.md
summary: "在 VR/AR 头显平台上驱动关节化化身，通常只能拿到稀疏运动信号——头显（HMD）+ 两个手柄的三点位姿。AvatarPoser 提出从这些稀疏运动感知实时重建全身关节姿态：基于 Transformer 的网络从稀疏输入预测全身关节旋转，并把全局运动与局部姿态解耦（用稳定的全局参考），再用逆运动学（IK）微调手臂，使预测末端精确对齐输入关节。AvatarPoser 在大规模 AMASS 动作数据上达 SOTA，为头显平台的化身驱动奠定基础（ECCV 2022）。"
---

# AvatarPoser

**AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

在 VR/AR 头显平台上驱动关节化化身，通常只能拿到稀疏运动信号——头显（HMD）+ 两个手柄的三点位姿。AvatarPoser 提出从这些稀疏运动感知实时重建全身关节姿态：基于 Transformer 的网络从稀疏输入预测全身关节旋转，并把全局运动与局部姿态解耦（用稳定的全局参考），再用逆运动学（IK）微调手臂，使预测末端精确对齐输入关节。AvatarPoser 在大规模 AMASS 动作数据上达 SOTA，为头显平台的化身驱动奠定基础（ECCV 2022）。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Sparse Motion Sensing | 稀疏运动感知（三点输入） |
| 3-point Input | 头显 + 两手柄三点位姿 |
| Articulated Pose | 关节化全身姿态 |
| Global/Local Decoupling | 全局运动与局部姿态解耦 |
| IK | Inverse Kinematics，逆运动学微调 |
| AMASS | 大规模人体动作数据集 |

## 为什么重要

- **从稀疏末端推全身**与人形"少传感器估全身状态"相通（状态估计/本仓 09）；
- **全局-局部解耦**对全身姿态/重心一致性有益（呼应 FRAME）；
- **IK 微调对齐末端**是把学习预测与硬约束结合的经典做法；
- 三点驱动化身的范式启发人形遥操作的稀疏输入控制。

## 解决什么问题

头显平台只有**三点稀疏信号**，却要驱动**全身化身**： - 从极稀疏输入推全身**欠约束**； - 要**实时**、手部要**精确对齐**输入； - 全局漂移影响稳定。

AvatarPoser 要：从三点稀疏输入实时、精确地重建全身关节姿态。

## 核心机制

1. **三点稀疏 → 全身姿态**：从 HMD + 双手柄重建关节化全身；
2. **Transformer + 全局-局部解耦**：稳定预测；
3. **IK 微调手臂**：末端精确对齐输入；
4. **AMASS SOTA**：头显化身驱动奠基（ECCV 2022）。

方法拆解（深读笔记小节）：Transformer 从三点预测全身关节；全局-局部解耦；IK 微调手臂（精确对齐）；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/AvatarPoser__Articulated_Full-Body_Pose_Tracking_from_Sparse_Motion_Sensing/AvatarPoser__Articulated_Full-Body_Pose_Tracking_from_Sparse_Motion_Sensing.html> |
| arXiv | <https://arxiv.org/abs/2207.13784> |
| 作者 | Jiaxi Jiang、Paul Streli、Huajian Qiu、Andreas Fender、Christian Holz 等（ETH Zürich SIPLAB） |
| 发表 | 2022 年 7 月（ECCV 2022） |
| 项目主页 | [siplab.org/projects/AvatarPoser](https://siplab.org/projects/AvatarPoser) · [code](https://github.com/eth-siplab/AvatarPoser) |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_avatarposer.md](../../sources/papers/humanoid_pnb_avatarposer.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/AvatarPoser__Articulated_Full-Body_Pose_Tracking_from_Sparse_Motion_Sensing/AvatarPoser__Articulated_Full-Body_Pose_Tracking_from_Sparse_Motion_Sensing.html>
- 论文：<https://arxiv.org/abs/2207.13784>

## 推荐继续阅读

- [机器人论文阅读笔记：AvatarPoser](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/AvatarPoser__Articulated_Full-Body_Pose_Tracking_from_Sparse_Motion_Sensing/AvatarPoser__Articulated_Full-Body_Pose_Tracking_from_Sparse_Motion_Sensing.html)
