---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2511.00041"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_endowing-gpt-4-with-a-humanoid-body.md
summary: "本文提出 BiBo 系统，让 GPT-4 这类视觉语言模型（VLM）直接控制人形机器人。与其收集海量训练数据，BiBo 利用 VLM 强大的开放世界泛化来降低数据采集需求。系统包含两部分：① 具身指令编译器（embodied instruction compiler）——把高层用户命令翻译成低层运动参数；② 基于扩散的运动执行器（diffusion-based motion executor）——生成对环境反馈自适应的拟人动作。结果：在开放环境的交互任务成功率 90.2%；文本引导的动作执行精度较此前方法提升 16.3%。"
---

# Endowing GPT-4 with a Humanoid Body

**Endowing GPT-4 with a Humanoid Body: Building the Bridge Between Off-the-Shelf VLMs and the Physical World** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

本文提出 BiBo 系统，让 GPT-4 这类视觉语言模型（VLM）直接控制人形机器人。与其收集海量训练数据，BiBo 利用 VLM 强大的开放世界泛化来降低数据采集需求。系统包含两部分：① 具身指令编译器（embodied instruction compiler）——把高层用户命令翻译成低层运动参数；② 基于扩散的运动执行器（diffusion-based motion executor）——生成对环境反馈自适应的拟人动作。结果：在开放环境的交互任务成功率 90.2%；文本引导的动作执行精度较此前方法提升 16.3%。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| BiBo | 本文系统名 |
| Off-the-shelf VLM | 现成视觉语言模型（如 GPT-4） |
| Instruction Compiler | 指令编译器，高层命令→低层参数 |
| Diffusion Executor | 扩散运动执行器 |
| Open-world Generalization | 开放世界泛化 |
| Adaptive Motion | 自适应（对环境反馈）动作 |

## 为什么重要

- **"现成 VLM + 轻量桥接"是低数据落地的诱人路线**：把通用模型能力借给具身；
- **编译器 + 扩散执行器**分工清晰：语义规划 vs 物理执行；
- **环境反馈自适应**是从"开环生成"走向"闭环可用"的关键；
- 与 SENTINEL、FRoM-W1 等语言-动作工作互为对照（端到端 vs 借现成 VLM）。

## 解决什么问题

让 VLM 控制人形通常需**海量具身数据**，成本高。论文问： - 能否**直接用现成 VLM**（不微调大数据）控制人形？ - 如何把 VLM 的**语义/规划**接到**低层物理动作**且**对环境自适应**？

BiBo 要：用现成 VLM 的开放世界泛化 + 轻量桥接，**少数据**地驱动人形。

## 核心机制

1. **现成 VLM 直接控人形**：借开放世界泛化，免大规模具身数据；
2. **具身指令编译器**：高层命令→低层运动参数；
3. **扩散运动执行器**：环境反馈自适应、拟人；
4. **强结果**：开放环境交互 90.2%、文本引导精度 +16.3%。

方法拆解（深读笔记小节）：具身指令编译器（高层→低层）；基于扩散的运动执行器（自适应拟人）；借 VLM 开放世界泛化降数据；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Endowing_GPT-4_with_a_Humanoid_Body__Bridge_Between_VLMs_and_the_Physical_World/Endowing_GPT-4_with_a_Humanoid_Body__Bridge_Between_VLMs_and_the_Physical_World.html> |
| arXiv | <https://arxiv.org/abs/2511.00041> |
| 作者 | Yingzhao Jian、Zhongan Wang、Yi Yang、Hehe Fan（浙江大学） |
| 发表 | 2025 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_endowing-gpt-4-with-a-humanoid-body.md](../../sources/papers/humanoid_pnb_endowing-gpt-4-with-a-humanoid-body.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Endowing_GPT-4_with_a_Humanoid_Body__Bridge_Between_VLMs_and_the_Physical_World/Endowing_GPT-4_with_a_Humanoid_Body__Bridge_Between_VLMs_and_the_Physical_World.html>
- 论文：<https://arxiv.org/abs/2511.00041>

## 推荐继续阅读

- [机器人论文阅读笔记：Endowing GPT-4 with a Humanoid Body](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Endowing_GPT-4_with_a_Humanoid_Body__Bridge_Between_VLMs_and_the_Physical_World/Endowing_GPT-4_with_a_Humanoid_Body__Bridge_Between_VLMs_and_the_Physical_World.html)
