---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2306.00378"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_example-based-motion-synthesis-via-generative-mo.md
summary: "GenMM 是一个生成模型，从单段或少量范例序列中\"挖（mine）\"出尽可能多样的动作。与现有数据驱动方法（通常需长时离线训练、易出视觉伪影、在大型复杂骨架上易失败）形成鲜明对比，GenMM 继承了知名 Motion Matching 方法的免训练特性与卓越质量。框架还可扩展到动作补全、关键帧引导生成、无限循环、动作重组等场景。核心是一个生成式动作匹配模块，用双向视觉相似度作代价函数，配多阶段框架从随机初始化逐步细化。它能在几分之一秒内合成高质量动作、处理复杂大型骨架、从极少范例生成多样动作。"
---

# Example-based Motion Synthesis via Generative Motion Matching

**Example-based Motion Synthesis via Generative Motion Matching** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

GenMM 是一个生成模型，从单段或少量范例序列中"挖（mine）"出尽可能多样的动作。与现有数据驱动方法（通常需长时离线训练、易出视觉伪影、在大型复杂骨架上易失败）形成鲜明对比，GenMM 继承了知名 Motion Matching 方法的免训练特性与卓越质量。框架还可扩展到动作补全、关键帧引导生成、无限循环、动作重组等场景。核心是一个生成式动作匹配模块，用双向视觉相似度作代价函数，配多阶段框架从随机初始化逐步细化。它能在几分之一秒内合成高质量动作、处理复杂大型骨架、从极少范例生成多样动作。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| GenMM | 生成式动作匹配模型 |
| Motion Matching | 经典免训练动作合成法 |
| Example-based | 范例驱动（单/少样本） |
| Bidirectional Similarity | 双向视觉相似度（代价函数） |
| Multi-stage | 多阶段逐步细化 |
| Training-free | 免训练 |

## 为什么重要

- **免训练范例驱动**对快速生成参考动作很实用，无需大数据/长训练；
- **双向相似度**是衡量动作质量的可迁移代价；
- **从少范例挖多样**契合人形稀缺动作数据的扩增需求；
- 动作重组/循环可为人形技能库提供多样参考。

## 解决什么问题

数据驱动动作合成的痛点： - **长时离线训练**； - **视觉伪影**； - **大型复杂骨架**上易失败； - 难从**极少范例**生成多样动作。

GenMM 要：**免训练**、高质量、可处理复杂骨架、从单/少范例生成多样动作。

## 核心机制

1. **GenMM 免训练生成式动作匹配**：继承 Motion Matching 质量；
2. **双向相似度 + 多阶段细化**：避伪影、稳健复杂骨架；
3. **极少范例 → 多样动作**：单/少样本；
4. **多场景 + 高效**：补全/关键帧/循环/重组，<1 秒。

方法拆解（深读笔记小节）：生成式动作匹配（双向相似度代价）；多阶段逐步细化；多场景扩展；效率；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Example-based_Motion_Synthesis_via_Generative_Motion_Matching/Example-based_Motion_Synthesis_via_Generative_Motion_Matching.html> |
| arXiv | <https://arxiv.org/abs/2306.00378> |
| 作者 | Weiyu Li、Xuelin Chen、Peizhuo Li、Olga Sorkine-Hornung、Baoquan Chen |
| 发表 | 2023 年 6 月 |
| 会议 | SIGGRAPH 2023 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_example-based-motion-synthesis-via-generative-mo.md](../../sources/papers/humanoid_pnb_example-based-motion-synthesis-via-generative-mo.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Example-based_Motion_Synthesis_via_Generative_Motion_Matching/Example-based_Motion_Synthesis_via_Generative_Motion_Matching.html>
- 论文：<https://arxiv.org/abs/2306.00378>

## 推荐继续阅读

- [机器人论文阅读笔记：Example-based Motion Synthesis via Generative Motion Matching](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/Example-based_Motion_Synthesis_via_Generative_Motion_Matching/Example-based_Motion_Synthesis_via_Generative_Motion_Matching.html)
