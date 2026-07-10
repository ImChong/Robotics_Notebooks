---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2505.11709"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_egodex.md
summary: "操作模仿学习有数据稀缺问题：不像语言/2D 视觉有互联网规模语料，灵巧操作没有。第一视角人类视频是被动可扩展的诱人来源，但现有大数据集（如 Ego4D）无原生手姿标注、也不聚焦物体操作。为此，作者用 Apple Vision Pro 采集 EgoDex ——迄今最大、最多样的人类灵巧操作数据集：829 小时第一视角视频，录制时即配 3D 手与手指跟踪（多台标定相机 + 机载 SLAM 精确跟踪每只手每个关节的位姿）。数据覆盖194 个桌面任务（从系鞋带到叠衣服）的多样操作行为。作者还在该数据上训练并系统评测用于手部轨迹预测的模仿学习策略，引入度量与基准。数据集公开下载。"
---

# EgoDex

**EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

操作模仿学习有数据稀缺问题：不像语言/2D 视觉有互联网规模语料，灵巧操作没有。第一视角人类视频是被动可扩展的诱人来源，但现有大数据集（如 Ego4D）无原生手姿标注、也不聚焦物体操作。为此，作者用 Apple Vision Pro 采集 EgoDex ——迄今最大、最多样的人类灵巧操作数据集：829 小时第一视角视频，录制时即配 3D 手与手指跟踪（多台标定相机 + 机载 SLAM 精确跟踪每只手每个关节的位姿）。数据覆盖194 个桌面任务（从系鞋带到叠衣服）的多样操作行为。作者还在该数据上训练并系统评测用于手部轨迹预测的模仿学习策略，引入度量与基准。数据集公开下载。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| EgoDex | 本文数据集 |
| Egocentric | 第一视角 |
| 3D Hand/Finger Tracking | 3D 手与手指跟踪 |
| SLAM | 同步定位与建图（机载） |
| Hand Trajectory Prediction | 手部轨迹预测任务 |
| Apple Vision Pro | 采集硬件 |

## 为什么重要

- **精确 3D 手姿是灵巧操作数据的金标准**，Vision Pro 让其规模化可行；
- **数据集 + 基准**是子领域进步的基础设施；
- 第一视角灵巧数据是人形操作的关键先验，与 Being-H0、H-RDT 等下游方法配套；
- Apple 等大厂入场，提示该方向的工业关注。

## 解决什么问题

灵巧操作缺**互联网规模数据**： - Ego4D 等大数据集**无手姿标注、不聚焦操作**； - 缺**带精确 3D 手姿**的大规模灵巧操作数据。

EgoDex 要：用 Vision Pro 采集**带 3D 手姿**的**大规模多样**灵巧操作数据集。

## 核心机制

1. **迄今最大灵巧操作数据集**：829h、194 任务、3D 手姿；
2. **录制即配精确手姿**：标定相机 + SLAM，免事后估计误差；
3. **手部轨迹预测基准**：度量 + IL 策略评测；
4. **开源**：推动机器人/视觉/基础模型。

方法拆解（深读笔记小节）：Apple Vision Pro 采集 + 精确手姿；规模与多样性；基准：手部轨迹预测；开源；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoDex__Learning_Dexterous_Manipulation_from_Large-Scale_Egocentric_Video/EgoDex__Learning_Dexterous_Manipulation_from_Large-Scale_Egocentric_Video.html> |
| arXiv | <https://arxiv.org/abs/2505.11709> |
| 作者 | Ryan Hoque、Peide Huang、David J. Yoon、Mouli Sivapurapu、Jian Zhang（Apple） |
| 发表 | 2025 年 5 月 |
| 源码 | [github.com/apple/ml-egodex](https://github.com/apple/ml-egodex) |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_egodex.md](../../sources/papers/humanoid_pnb_egodex.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoDex__Learning_Dexterous_Manipulation_from_Large-Scale_Egocentric_Video/EgoDex__Learning_Dexterous_Manipulation_from_Large-Scale_Egocentric_Video.html>
- 论文：<https://arxiv.org/abs/2505.11709>

## 推荐继续阅读

- [机器人论文阅读笔记：EgoDex](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/EgoDex__Learning_Dexterous_Manipulation_from_Large-Scale_Egocentric_Video/EgoDex__Learning_Dexterous_Manipulation_from_Large-Scale_Egocentric_Video.html)
