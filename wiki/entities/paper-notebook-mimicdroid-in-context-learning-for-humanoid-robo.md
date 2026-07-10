---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2509.09769"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_mimicdroid.md
summary: "目标是让人形从少量视频示例高效解决新操作任务。上下文学习（ICL）因测试时数据高效、快速适应而有前景，但现有 ICL 方法依赖费力的遥操作数据，难规模化。本文用人类玩耍视频（human play videos）——人们自由与环境交互的连续、无标注视频——作为可扩展、多样的训练源。提出 MimicDroid：仅用人类玩耍视频做训练，抽取行为相似的轨迹对，训练策略以一条轨迹为条件预测另一条的动作，从而获得测试时适应新物体/环境的 ICL 能力。为弥合具身差距，先用运动学相似性把 RGB 视频估计的人手腕姿态重定向到人形；训练时随机块遮挡（patch masking）降低对人类特有线索的过拟合、增强对视觉差异的鲁棒。作者还提出一个开源仿真基准（难度递增）评估少样本学习；MimicDroid 优于 SOTA，真机成功率近两倍。"
---

# MimicDroid

**MimicDroid: In-Context Learning for Humanoid Robot Manipulation from Human Play Videos** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

目标是让人形从少量视频示例高效解决新操作任务。上下文学习（ICL）因测试时数据高效、快速适应而有前景，但现有 ICL 方法依赖费力的遥操作数据，难规模化。本文用人类玩耍视频（human play videos）——人们自由与环境交互的连续、无标注视频——作为可扩展、多样的训练源。提出 MimicDroid：仅用人类玩耍视频做训练，抽取行为相似的轨迹对，训练策略以一条轨迹为条件预测另一条的动作，从而获得测试时适应新物体/环境的 ICL 能力。为弥合具身差距，先用运动学相似性把 RGB 视频估计的人手腕姿态重定向到人形；训练时随机块遮挡（patch masking）降低对人类特有线索的过拟合、增强对视觉差异的鲁棒。作者还提出一个开源仿真基准（难度递增）评估少样本学习；MimicDroid 优于 SOTA，真机成功率近两倍。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| ICL | In-Context Learning，上下文学习 |
| Play Video | 玩耍视频，自由交互的无标注视频 |
| Trajectory Pair | 轨迹对，行为相似的两段 |
| Retargeting | 重定向（人手腕→人形） |
| Patch Masking | 随机块遮挡，防过拟合 |
| Few-shot | 少样本 |

## 为什么重要

- **人类玩耍视频是海量免费的 ICL 训练源**，比遥操作更可扩展；
- **ICL 让人形"看几个例子就会"**，是快速适应的诱人范式；
- **随机块遮挡**是缓解人-机视觉差异过拟合的简单有效技巧；
- 与 In-N-On、Masquerade 等"从人类视频学操作"路线互补。

## 解决什么问题

让人形**少样本快速学新任务**： - ICL 有前景，但**依赖遥操作数据**，难规模化； - 想用**人类玩耍视频**（海量、无标注），但有**具身差距**与**人类特有线索过拟合**。

MimicDroid 要：**仅用人类玩耍视频**训练出有 ICL 能力的人形操作策略。

## 核心机制

1. **仅用人类玩耍视频的 ICL**：摆脱对遥操作数据的依赖；
2. **轨迹对条件预测**：获得测试时少样本适应能力；
3. **重定向 + 块遮挡**：弥合具身差距、防过拟合；
4. **开源基准 + 真机≈2×SOTA**。

方法拆解（深读笔记小节）：从玩耍视频抽轨迹对、学 ICL；重定向人手腕姿态；随机块遮挡防过拟合；基准与结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos.html> |
| arXiv | <https://arxiv.org/abs/2509.09769> |
| 作者 | Rutav Shah、Shuijing Liu、Zhenyu Jiang、Mingyo Seo、Roberto Martín-Martín、Yuke Zhu（UT Austin） |
| 发表 | 2025 年 9 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_mimicdroid.md](../../sources/papers/humanoid_pnb_mimicdroid.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos.html>
- 论文：<https://arxiv.org/abs/2509.09769>

## 推荐继续阅读

- [机器人论文阅读笔记：MimicDroid](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos/MimicDroid__In-Context_Learning_for_Humanoid_Manipulation_from_Human_Play_Videos.html)
