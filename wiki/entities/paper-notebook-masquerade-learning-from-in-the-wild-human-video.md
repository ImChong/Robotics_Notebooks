---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2508.09976"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_masquerade.md
summary: "机器人操作仍数据稀缺——最大的机器人数据集也比驱动语言/视觉突破的数据小几个数量级。Masquerade 通过编辑野外第一视角人类视频来闭合人-机视觉具身差距，再用编辑后的视频学机器人策略。流程把每段人类视频变成机器人化演示：① 估计 3D 手姿；② 修复涂抹（inpaint）人臂；③ 叠加渲染的双臂机器人，使其跟踪恢复的末端轨迹。在 67.5 万帧编辑片段上预训练视觉编码器以预测未来 2D 机器人关键点，并在每任务仅 50 条机器人演示上微调扩散策略头（继续保留该辅助损失），所得策略泛化显著更好。在三个长时程双手厨房任务、各三个未见场景上，Masquerade 较基线高 5–6 倍；消融显示机器人叠加与协同训练都不可或缺，性能随编辑人类视频量对数增长。"
---

# Masquerade

**Masquerade: Learning from In-the-wild Human Videos using Data-Editing** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

机器人操作仍数据稀缺——最大的机器人数据集也比驱动语言/视觉突破的数据小几个数量级。Masquerade 通过编辑野外第一视角人类视频来闭合人-机视觉具身差距，再用编辑后的视频学机器人策略。流程把每段人类视频变成机器人化演示：① 估计 3D 手姿；② 修复涂抹（inpaint）人臂；③ 叠加渲染的双臂机器人，使其跟踪恢复的末端轨迹。在 67.5 万帧编辑片段上预训练视觉编码器以预测未来 2D 机器人关键点，并在每任务仅 50 条机器人演示上微调扩散策略头（继续保留该辅助损失），所得策略泛化显著更好。在三个长时程双手厨房任务、各三个未见场景上，Masquerade 较基线高 5–6 倍；消融显示机器人叠加与协同训练都不可或缺，性能随编辑人类视频量对数增长。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Data-Editing | 数据编辑，把人类视频改造成机器人演示 |
| Inpainting | 图像修复，涂掉人臂 |
| Robot Overlay | 叠加渲染机器人 |
| Co-training | 协同训练（辅助损失 + 策略） |
| 2D Keypoints | 2D 机器人关键点 |
| Visual Embodiment Gap | 视觉具身差距 |

## 为什么重要

- **"把人类视频改造成机器人样子"是用好人类数据的关键洞见**：视觉一致才好学；
- **机器人叠加 + 协同训练缺一不可**，提示视觉对齐与辅助监督的协同；
- **少量机器人演示 + 海量编辑视频**是高性价比配方；
- 与 MimicDroid、In-N-On 等共同推进从人类视频学操作。

## 解决什么问题

机器人数据稀缺，人类视频海量但有**视觉具身差距**（看起来是人手不是机器人）： - 直接学人类视频，策略看到的视觉与机器人不一致； - 需要**闭合视觉差距**才能用好人类视频。

Masquerade 要：用**数据编辑**把人类视频"机器人化"，从而利用海量人类视频。

## 核心机制

1. **数据编辑闭合视觉具身差距**：手姿估计 + 涂臂 + 机器人叠加；
2. **预训练 + 协同训练**：67.5 万帧预测 2D 关键点，仅 50 演示/任务微调；
3. **显著泛化**：双手厨房任务较基线 5–6 倍；
4. **可扩展**：性能随编辑视频量对数增长。

方法拆解（深读笔记小节）：三步数据编辑（人类视频 → 机器人演示）；预训练 + 协同训练；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Masquerade__Learning_from_In-the-wild_Human_Videos_using_Data-Editing/Masquerade__Learning_from_In-the-wild_Human_Videos_using_Data-Editing.html> |
| arXiv | <https://arxiv.org/abs/2508.09976> |
| 作者 | Marion Lepert、Jiaying Fang、Jeannette Bohg（Stanford） |
| 发表 | 2025 年 8 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_masquerade.md](../../sources/papers/humanoid_pnb_masquerade.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Masquerade__Learning_from_In-the-wild_Human_Videos_using_Data-Editing/Masquerade__Learning_from_In-the-wild_Human_Videos_using_Data-Editing.html>
- 论文：<https://arxiv.org/abs/2508.09976>

## 推荐继续阅读

- [机器人论文阅读笔记：Masquerade](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Masquerade__Learning_from_In-the-wild_Human_Videos_using_Data-Editing/Masquerade__Learning_from_In-the-wild_Human_Videos_using_Data-Editing.html)
