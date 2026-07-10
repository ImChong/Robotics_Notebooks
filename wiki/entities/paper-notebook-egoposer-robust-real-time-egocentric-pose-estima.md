---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2308.06493"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_egoposer.md
summary: "仅用头与手位姿做全身第一视角姿态估计是头显平台驱动化身的活跃方向，但现有方法过度依赖室内动捕空间（数据录制环境），且假设关节连续跟踪与统一体型。EgoPoser 在更真实的设定下做到鲁棒：手部位置/朝向只有进入头显视野（FoV）时才被跟踪——即稀疏且间歇的观测。它还有三点关键贡献：① 在间歇手部跟踪下仍鲁棒建模全身姿态；② 全局运动分解——独立于全局位置预测全身姿态；③ 高效的 SlowFast 模块设计，在捕捉更长运动时序的同时保持算力高效；并能跨不同用户体型泛化。ECCV 2024。"
---

# EgoPoser

**EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

仅用头与手位姿做全身第一视角姿态估计是头显平台驱动化身的活跃方向，但现有方法过度依赖室内动捕空间（数据录制环境），且假设关节连续跟踪与统一体型。EgoPoser 在更真实的设定下做到鲁棒：手部位置/朝向只有进入头显视野（FoV）时才被跟踪——即稀疏且间歇的观测。它还有三点关键贡献：① 在间歇手部跟踪下仍鲁棒建模全身姿态；② 全局运动分解——独立于全局位置预测全身姿态；③ 高效的 SlowFast 模块设计，在捕捉更长运动时序的同时保持算力高效；并能跨不同用户体型泛化。ECCV 2024。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Egocentric | 第一视角（头显） |
| Intermittent | 间歇（手出视野则丢跟踪） |
| FoV | Field of View，头显视野 |
| Global Motion Decomposition | 全局运动分解（独立于全局位置） |
| SlowFast | 慢-快双路时序模块 |
| Body Shape Generalization | 跨体型泛化 |

## 为什么重要

- **"间歇/缺失观测下鲁棒"是真实部署的关键**，与人形状态估计在传感缺失时的鲁棒诉求一致；
- **全局运动分解**对全身状态估计的泛化有借鉴（呼应 AvatarPoser、FRAME）；
- **SlowFast 长时序 + 高效**是实时全身估计的实用结构；
- 稀疏感知驱动全身，对人形遥操作/化身有直接价值。

## 解决什么问题

现有第一视角全身姿态估计**假设太理想**： - 依赖**室内动捕空间**、**连续**关节跟踪、**统一体型**； - 真实中手**常出视野**→**间歇**观测； - 全局位置变化影响稳定。

EgoPoser 要：在**稀疏间歇**观测、**任意场景、任意体型**下鲁棒实时估计全身姿态。

## 核心机制

1. **间歇观测鲁棒**：手出视野也能稳健估全身姿态；
2. **全局运动分解**：独立于全局位置预测，提升泛化；
3. **SlowFast 模块**：长时序 + 算力高效；
4. **跨体型泛化**：摆脱统一体型假设（ECCV 2024）。

方法拆解（深读笔记小节）：间歇手部观测下的鲁棒建模；全局运动分解（独立于全局位置）；SlowFast 模块（长时序 + 高效）；跨体型泛化；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/EgoPoser__Robust_Real-Time_Egocentric_Pose_Estimation_from_Sparse_Sensors/EgoPoser__Robust_Real-Time_Egocentric_Pose_Estimation_from_Sparse_Sensors.html> |
| arXiv | <https://arxiv.org/abs/2308.06493> |
| 作者 | Jiaxi Jiang、Paul Streli、Manuel Meier、Christian Holz（ETH Zürich SIPLAB） |
| 发表 | 2023 年 8 月（ECCV 2024） |
| 项目主页 | [siplab.org/projects/EgoPoser](https://siplab.org/projects/EgoPoser) · [code](https://github.com/eth-siplab/EgoPoser) |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_egoposer.md](../../sources/papers/humanoid_pnb_egoposer.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/EgoPoser__Robust_Real-Time_Egocentric_Pose_Estimation_from_Sparse_Sensors/EgoPoser__Robust_Real-Time_Egocentric_Pose_Estimation_from_Sparse_Sensors.html>
- 论文：<https://arxiv.org/abs/2308.06493>

## 推荐继续阅读

- [机器人论文阅读笔记：EgoPoser](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/EgoPoser__Robust_Real-Time_Egocentric_Pose_Estimation_from_Sparse_Sensors/EgoPoser__Robust_Real-Time_Egocentric_Pose_Estimation_from_Sparse_Sensors.html)
