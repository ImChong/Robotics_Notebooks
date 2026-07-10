---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2503.21268"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_climbingcap.md
summary: "人体动作恢复（HMR）研究多聚焦地面动作（如跑步），对离地（off-ground）的攀岩动作研究稀少，部分因攀岩动作数据集（尤其大规模、有挑战性的 3D 标注）匮乏。作者采集 AscendMotion ——一个大规模、标注良好、有挑战性的攀岩动作数据集：41.2 万帧 RGB、LiDAR 帧与 IMU 测量，含 22 位熟练攀岩教练在 12 面不同岩壁上的攀岩动作。攀岩动作捕捉难在需精确恢复复杂姿态 + 全局位置；现有全局 HMR 方法难以胜任。为此提出 ClimbingCap，在全局坐标系下连续重建 3D 攀岩动作：关键是用 RGB 与 LiDAR 分别在相机坐标与全局坐标重建动作，并联合优化。CVPR 2025 收录。"
---

# ClimbingCap

**ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

人体动作恢复（HMR）研究多聚焦地面动作（如跑步），对离地（off-ground）的攀岩动作研究稀少，部分因攀岩动作数据集（尤其大规模、有挑战性的 3D 标注）匮乏。作者采集 AscendMotion ——一个大规模、标注良好、有挑战性的攀岩动作数据集：41.2 万帧 RGB、LiDAR 帧与 IMU 测量，含 22 位熟练攀岩教练在 12 面不同岩壁上的攀岩动作。攀岩动作捕捉难在需精确恢复复杂姿态 + 全局位置；现有全局 HMR 方法难以胜任。为此提出 ClimbingCap，在全局坐标系下连续重建 3D 攀岩动作：关键是用 RGB 与 LiDAR 分别在相机坐标与全局坐标重建动作，并联合优化。CVPR 2025 收录。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| HMR | Human Motion Recovery，人体动作恢复 |
| AscendMotion | 本文攀岩多模态数据集 |
| RGB / LiDAR / IMU | 彩色 / 激光雷达 / 惯性 |
| Off-ground | 离地动作（攀岩） |
| World Coordinate | 世界（全局）坐标系 |
| Global Position | 全局位置 |

## 为什么重要

- **离地/强接触动作（攀岩）**是高难全身动作，对人形多接触/跑酷有数据价值；
- **RGB + LiDAR 分坐标 + 联合优化**是世界坐标全身重建的实用方案；
- **全局位置恢复**对人形 loco-manip 的世界系跟踪相关（呼应 HiWET）；
- 攀岩这类极限动作可作为人形高难技能的参考动作源。

## 解决什么问题

攀岩这类**离地动作**的动捕被忽视： - 缺**大规模 3D 标注**攀岩数据； - 需同时恢复**复杂姿态 + 全局位置**； - 现有全局 HMR 方法难胜任。

ClimbingCap 要：建数据集（AscendMotion）+ 方法，在世界坐标下连续重建攀岩动作。

## 核心机制

1. **AscendMotion 攀岩数据集**：41.2 万帧多模态，22 教练/12 岩壁；
2. **ClimbingCap 方法**：RGB（相机）+ LiDAR（全局）分坐标重建 + 联合优化；
3. **世界坐标连续重建**：姿态 + 全局位置；
4. **离地动作**：填补攀岩动捕空白（CVPR 2025）。

方法拆解（深读笔记小节）：AscendMotion 多模态数据集；ClimbingCap：RGB + LiDAR 分坐标重建 + 联合优化；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/ClimbingCap__Multi-Modal_Dataset_and_Method_for_Rock_Climbing_in_World_Coordinate/ClimbingCap__Multi-Modal_Dataset_and_Method_for_Rock_Climbing_in_World_Coordinate.html> |
| arXiv | <https://arxiv.org/abs/2503.21268> |
| 作者 | Ming Yan、Xincheng Lin、Yudi Dai、Yuexin Ma、Lan Xu、Chenglu Wen、Siqi Shen、Cheng Wang（厦大 / 上科大等） |
| 发表 | 2025 年 3 月 |
| 会议 | CVPR 2025 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_climbingcap.md](../../sources/papers/humanoid_pnb_climbingcap.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/ClimbingCap__Multi-Modal_Dataset_and_Method_for_Rock_Climbing_in_World_Coordinate/ClimbingCap__Multi-Modal_Dataset_and_Method_for_Rock_Climbing_in_World_Coordinate.html>
- 论文：<https://arxiv.org/abs/2503.21268>

## 推荐继续阅读

- [机器人论文阅读笔记：ClimbingCap](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/ClimbingCap__Multi-Modal_Dataset_and_Method_for_Rock_Climbing_in_World_Coordinate/ClimbingCap__Multi-Modal_Dataset_and_Method_for_Rock_Climbing_in_World_Coordinate.html)
