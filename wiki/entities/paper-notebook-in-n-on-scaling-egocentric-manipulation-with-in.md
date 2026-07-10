---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2511.15704"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_in-n-on.md
summary: "第一视角（egocentric）视频是学操作策略的宝贵可扩展数据源，但数据异质性大，多数方法只把人类数据用于简单预训练，没释放全部潜力。本文先给出一套可扩展配方：把人类数据分成两类——野外（in-the-wild）与任务对齐（on-task），并系统分析如何使用。作者整理出数据集 PHSD，含 1000+ 小时多样野外第一视角数据与 20+ 小时直接对齐目标任务的任务数据。据此训练一个大型语言条件流匹配策略 Human0；配合域适应技术，Human0 缩小人到人形的差距。实证表明，规模化人类数据带来若干新性质：仅凭人类数据就能听从语言指令、少样本学习、以及用任务数据提升的鲁棒性。"
---

# In-N-On

**In-N-On: Scaling Egocentric Manipulation with in-the-wild and on-task Data** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

第一视角（egocentric）视频是学操作策略的宝贵可扩展数据源，但数据异质性大，多数方法只把人类数据用于简单预训练，没释放全部潜力。本文先给出一套可扩展配方：把人类数据分成两类——野外（in-the-wild）与任务对齐（on-task），并系统分析如何使用。作者整理出数据集 PHSD，含 1000+ 小时多样野外第一视角数据与 20+ 小时直接对齐目标任务的任务数据。据此训练一个大型语言条件流匹配策略 Human0；配合域适应技术，Human0 缩小人到人形的差距。实证表明，规模化人类数据带来若干新性质：仅凭人类数据就能听从语言指令、少样本学习、以及用任务数据提升的鲁棒性。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Egocentric | 第一视角（头戴视角） |
| In-the-wild / On-task | 野外 / 任务对齐数据 |
| PHSD | 本文数据集（1000h 野外 + 20h 任务） |
| Human0 | 语言条件流匹配策略 |
| Flow Matching | 流匹配生成策略 |
| Domain Adaptation | 域适应，缩小人↔人形差距 |

## 为什么重要

- **"野外 + 任务对齐"二分**是用好海量人类数据的关键洞见：规模来自野外、对齐来自少量任务数据；
- **语言条件 + 流匹配**让策略可指令驱动且高效；
- **域适应**是人到人形落地的必备环节；
- 与 Dexterity from Smart Lenses、EgoDex 等共同推进第一视角数据规模化。

## 解决什么问题

第一视角人类数据潜力大但用不好： - **异质性大**，多数只做**简单预训练**； - 缺**如何分类与使用**数据的系统配方； - 人到人形有**域差距**。

In-N-On 要：一套**可扩展配方**（野外 + 任务对齐）+ 数据集 + 策略，释放第一视角数据潜力。

## 核心机制

1. **可扩展第一视角数据配方**：野外 + 任务对齐两类 + 使用分析；
2. **PHSD 数据集**：1000h 野外 + 20h 任务对齐；
3. **Human0 语言条件流匹配 + 域适应**：缩小人↔人形差距；
4. **涌现新性质**：仅人类数据听指令、少样本、任务数据增鲁棒。

方法拆解（深读笔记小节）：数据分类：野外 + 任务对齐；PHSD 数据集；Human0：语言条件流匹配 + 域适应；涌现新性质；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data.html> |
| arXiv | <https://arxiv.org/abs/2511.15704> |
| 作者 | Xiongyi Cai、Ri-Zhao Qiu、Geng Chen、Lai Wei、Tianshu Huang、Xuxin Cheng、Xiaolong Wang（UC San Diego） |
| 发表 | 2025 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_in-n-on.md](../../sources/papers/humanoid_pnb_in-n-on.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data.html>
- 论文：<https://arxiv.org/abs/2511.15704>

## 推荐继续阅读

- [机器人论文阅读笔记：In-N-On](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data/In-N-On__Scaling_Egocentric_Manipulation_with_in-the-wild_and_on-task_Data.html)
