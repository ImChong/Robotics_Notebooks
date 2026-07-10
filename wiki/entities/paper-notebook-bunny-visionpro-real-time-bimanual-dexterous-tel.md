---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2407.03162"
related:
  - ../overview/paper-notebook-category-07-teleoperation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_bunny-visionpro.md
summary: "遥操作是采集人类演示的关键工具，但用双手灵巧手控制机器人很难——以往系统难以协调两只手做精细操作。Bunny-VisionPro 是一个实时双手灵巧遥操作系统，基于 VR 头显。不同于以往纯视觉方案，作者设计了低成本设备给操作者力触觉反馈，增强沉浸感。系统通过内置碰撞规避与奇异点规避优先保证安全，同时以创新设计保持实时。在标准任务集上，Bunny-VisionPro 成功率更高、用时更短；其高质量演示提升下游模仿学习的表现与泛化；尤其首次支持具有挑战性的多阶段、长时程双手灵巧操作任务的模仿学习。"
---

# Bunny-VisionPro

**Bunny-VisionPro: Real-Time Bimanual Dexterous Teleoperation for Imitation Learning** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：07_Teleoperation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

遥操作是采集人类演示的关键工具，但用双手灵巧手控制机器人很难——以往系统难以协调两只手做精细操作。Bunny-VisionPro 是一个实时双手灵巧遥操作系统，基于 VR 头显。不同于以往纯视觉方案，作者设计了低成本设备给操作者力触觉反馈，增强沉浸感。系统通过内置碰撞规避与奇异点规避优先保证安全，同时以创新设计保持实时。在标准任务集上，Bunny-VisionPro 成功率更高、用时更短；其高质量演示提升下游模仿学习的表现与泛化；尤其首次支持具有挑战性的多阶段、长时程双手灵巧操作任务的模仿学习。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Bimanual | 双手 |
| Dexterous | 灵巧（多指手操作） |
| Haptic Feedback | 力触觉反馈 |
| Singularity Avoidance | 奇异点规避 |
| Collision Avoidance | 碰撞规避 |
| Imitation Learning | 模仿学习 |

## 为什么重要

- **力触觉反馈显著提升灵巧遥操作质量**：操作者"感受到"接触才能做精细任务；
- **安全规避内建**让遥操作既快又稳，利于规模化采集；
- **长时程多阶段双手演示**是稀缺且珍贵的数据，对复杂操作策略至关重要；
- 与 ACE、TeleOpBench 等共同构成灵巧遥操作的数据/系统生态。

## 解决什么问题

双手灵巧遥操作难点： - **协调两只灵巧手**做精细操作难； - 纯视觉方案**缺力反馈**、沉浸差； - 要兼顾**安全**（防碰撞/奇异）与**实时**。

论文要：一个**实时、带力反馈、安全**的双手灵巧遥操作系统，并产出高质量演示供模仿学习。

## 核心机制

1. **实时双手灵巧遥操作（VR）**：协调两只灵巧手做精细操作；
2. **低成本力触觉反馈**：比纯视觉更沉浸；
3. **安全 + 实时**：碰撞与奇异点规避；
4. **促进模仿学习**：高质量演示提升泛化，首次支持多阶段长时程双手 IL。

方法拆解（深读笔记小节）：VR 头显 + 低成本力触觉反馈；安全：碰撞 + 奇异点规避；高质量演示 → 模仿学习；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 07_Teleoperation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Bunny-VisionPro__Real-Time_Bimanual_Dexterous_Teleoperation_for_Imitation_Learning/Bunny-VisionPro__Real-Time_Bimanual_Dexterous_Teleoperation_for_Imitation_Learning.html> |
| arXiv | <https://arxiv.org/abs/2407.03162> |
| 作者 | Runyu Ding、Yuzhe Qin、Jiyue Zhu、Chengzhe Jia、Shiqi Yang、Ruihan Yang、Xiaojuan Qi、Xiaolong Wang（HKU / UCSD） |
| 发表 | 2024 年 7 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-07-teleoperation](../overview/paper-notebook-category-07-teleoperation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_bunny-visionpro.md](../../sources/papers/humanoid_pnb_bunny-visionpro.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Bunny-VisionPro__Real-Time_Bimanual_Dexterous_Teleoperation_for_Imitation_Learning/Bunny-VisionPro__Real-Time_Bimanual_Dexterous_Teleoperation_for_Imitation_Learning.html>
- 论文：<https://arxiv.org/abs/2407.03162>

## 推荐继续阅读

- [机器人论文阅读笔记：Bunny-VisionPro](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/Bunny-VisionPro__Real-Time_Bimanual_Dexterous_Teleoperation_for_Imitation_Learning/Bunny-VisionPro__Real-Time_Bimanual_Dexterous_Teleoperation_for_Imitation_Learning.html)
