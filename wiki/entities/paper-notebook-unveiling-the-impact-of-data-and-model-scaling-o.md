---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2511.09241"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_unveiling-the-impact-of-data-and-model-scaling-o.md
summary: "数据规模一直是机器人学习的瓶颈。对人形而言，人类视频与动作数据海量、免费、易得，且其语义可用于模态对齐与高层控制学习。但如何挖原始视频、抽出机器人可学的表示、并用于可扩展学习仍是开放问题。为此，作者用一条自动化流水线造出 Humanoid-Union ——一个260+ 小时、多样高质量、带语义标注（源自人类动作视频）的人形动作数据集，并可经同一流水线继续扩展。在此数据上，提出 SCHUR 可扩展学习框架，系统研究大规模数据对人形高层控制的影响。结果：相比此前方法，重建 MPJPE 提升 37%、文本-动作对齐 FID 提升 25%。"
---

# Unveiling the Impact of Data and Model Scaling on High-Level Control for Humanoid Robots

**Unveiling the Impact of Data and Model Scaling on High-Level Control for Humanoid Robots** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

数据规模一直是机器人学习的瓶颈。对人形而言，人类视频与动作数据海量、免费、易得，且其语义可用于模态对齐与高层控制学习。但如何挖原始视频、抽出机器人可学的表示、并用于可扩展学习仍是开放问题。为此，作者用一条自动化流水线造出 Humanoid-Union ——一个260+ 小时、多样高质量、带语义标注（源自人类动作视频）的人形动作数据集，并可经同一流水线继续扩展。在此数据上，提出 SCHUR 可扩展学习框架，系统研究大规模数据对人形高层控制的影响。结果：相比此前方法，重建 MPJPE 提升 37%、文本-动作对齐 FID 提升 25%。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| High-Level Control | 高层控制，语义/意图层面的控制 |
| Humanoid-Union | 本文 260+ 小时带语义标注的数据集 |
| SCHUR | 本文提出的可扩展学习框架 |
| Data Scaling | 数据规模化 |
| MPJPE | 每关节平均位置误差，越低越好 |
| FID | 衡量生成质量/对齐的距离指标 |

## 为什么重要

- **"规模化研究"本身是贡献**：把"数据/模型规模 → 性能"做成可量化对象，指导后续投入；
- **语义标注是高层控制的关键**：让控制能被语言/意图驱动，呼应 FRoM-W1、SENTINEL 等语言-动作工作；
- **自动化数据流水线**是人形数据飞轮的引擎，与 SUGAR、UniAct 的数据观一致；
- **260+ 小时**的体量在人形动作领域可观，利于探究 scaling law。

## 解决什么问题

人形高层控制的核心瓶颈是**数据**： - 人类视频/动作虽**海量免费**，但**如何挖掘、抽取机器人可学表示、做可扩展学习**没解决； - 还缺**带语义标注**的大规模人形动作数据来支撑高层（语义）控制。

论文要：① 造出**大规模带语义**的人形动作数据；② 用它**系统揭示数据/模型规模**对高层控制的影响。

## 核心机制

1. **Humanoid-Union 大规模数据集**：260+ 小时、带语义、可扩展，源自人类视频；
2. **SCHUR 可扩展学习框架**：系统研究数据/模型规模对高层控制的影响；
3. **自动化数据流水线**：把免费人类视频转为机器人可学的带语义动作；
4. **量化提升**：重建 MPJPE +37%、对齐 FID +25%，真机验证。

方法拆解（深读笔记小节）：Humanoid-Union 数据集（自动化流水线）；SCHUR 可扩展学习框架；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Unveiling_the_Impact_of_Data_and_Model_Scaling_on_High-Level_Control_for_Humanoid/Unveiling_the_Impact_of_Data_and_Model_Scaling_on_High-Level_Control_for_Humanoid.html> |
| arXiv | <https://arxiv.org/abs/2511.09241> |
| 作者 | Yuxi Wei、Zirui Wang、Kangning Yin、Yue Hu、Jingbo Wang、Siheng Chen（上交大 / 上海 AI Lab 等） |
| 发表 | 2025 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_unveiling-the-impact-of-data-and-model-scaling-o.md](../../sources/papers/humanoid_pnb_unveiling-the-impact-of-data-and-model-scaling-o.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Unveiling_the_Impact_of_Data_and_Model_Scaling_on_High-Level_Control_for_Humanoid/Unveiling_the_Impact_of_Data_and_Model_Scaling_on_High-Level_Control_for_Humanoid.html>
- 论文：<https://arxiv.org/abs/2511.09241>

## 推荐继续阅读

- [机器人论文阅读笔记：Unveiling the Impact of Data and Model Scaling on High-Level Control for Humanoid Robots](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Unveiling_the_Impact_of_Data_and_Model_Scaling_on_High-Level_Control_for_Humanoid/Unveiling_the_Impact_of_Data_and_Model_Scaling_on_High-Level_Control_for_Humanoid.html)
