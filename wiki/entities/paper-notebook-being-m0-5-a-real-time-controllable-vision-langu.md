---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2508.07863"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_being-m0-5.md
summary: "人类动作生成潜力巨大，但现有视觉-语言-动作模型（VLMM）实用部署受限。作者指出可控性是主瓶颈，体现在五方面：对多样人类指令响应不足、姿态初始化能力有限、长序列表现差、对未见场景处理不足、缺乏对各身体部位的细粒度控制。为此提出Being-M0.5，并引入 HuMo100M ——迄今最大最全的人类动作数据集（500 万+ 自采动作序列、1 亿条多任务指令实例、细粒度部位级标注）。方法用部位感知残差量化（part-aware residual quantization）做动作 token 化，实现逐部位的精细控制。模型在多个动作生成基准上达 SOTA，同时保持实时执行效率。"
---

# Being-M0.5

**Being-M0.5: A Real-Time Controllable Vision-Language-Motion Model** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

人类动作生成潜力巨大，但现有视觉-语言-动作模型（VLMM）实用部署受限。作者指出可控性是主瓶颈，体现在五方面：对多样人类指令响应不足、姿态初始化能力有限、长序列表现差、对未见场景处理不足、缺乏对各身体部位的细粒度控制。为此提出Being-M0.5，并引入 HuMo100M ——迄今最大最全的人类动作数据集（500 万+ 自采动作序列、1 亿条多任务指令实例、细粒度部位级标注）。方法用部位感知残差量化（part-aware residual quantization）做动作 token 化，实现逐部位的精细控制。模型在多个动作生成基准上达 SOTA，同时保持实时执行效率。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| VLMM | Vision-Language-Motion Model |
| HuMo100M | 本文大规模人类动作数据集 |
| Part-aware Residual Quantization | 部位感知残差量化（动作 token 化） |
| Controllability | 可控性（本文主攻瓶颈） |
| Pose Initialization | 姿态初始化 |
| Part-level | 部位级（逐身体部位控制） |

## 为什么重要

- **逐部位细粒度控制**对人形全身动作（上身操作 + 下身行走分控）有借鉴；
- **大规模动作数据 + 量化 token 化**是动作生成模型的主流配方，可迁移到人形动作生成（FRoM-W1、UniAct 等）；
- **可控性五维度**是评估动作生成是否实用的好框架；
- 动作生成是人形"语言→动作"上游的关键一环。

## 解决什么问题

VLMM 的**可控性**不足（五大短板）：响应多样指令差、姿态初始化弱、长序列差、未见场景差、**缺逐部位细粒度控制**。论文要：一个**实时、可控**、能逐部位控制的 VLMM，并配足够大的数据。

## 核心机制

1. **诊断 VLMM 可控性五大短板**；
2. **HuMo100M**：迄今最大最全人类动作数据集（500 万+/1 亿/部位标注）；
3. **部位感知残差量化**：逐部位细粒度控制；
4. **实时 SOTA**：多基准领先且实时。

方法拆解（深读笔记小节）：HuMo100M 大规模数据集；部位感知残差量化（逐部位控制）；实时可控；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Being-M0.5__A_Real-Time_Controllable_Vision-Language-Motion_Model/Being-M0.5__A_Real-Time_Controllable_Vision-Language-Motion_Model.html> |
| arXiv | <https://arxiv.org/abs/2508.07863> |
| 作者 | Bin Cao、Sipeng Zheng、Ye Wang、Qin Jin、Jing Liu、Zongqing Lu 等（BAAI / 人大等） |
| 发表 | 2025 年 8 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_being-m0-5.md](../../sources/papers/humanoid_pnb_being-m0-5.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Being-M0.5__A_Real-Time_Controllable_Vision-Language-Motion_Model/Being-M0.5__A_Real-Time_Controllable_Vision-Language-Motion_Model.html>
- 论文：<https://arxiv.org/abs/2508.07863>

## 推荐继续阅读

- [机器人论文阅读笔记：Being-M0.5](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Being-M0.5__A_Real-Time_Controllable_Vision-Language-Motion_Model/Being-M0.5__A_Real-Time_Controllable_Vision-Language-Motion_Model.html)
