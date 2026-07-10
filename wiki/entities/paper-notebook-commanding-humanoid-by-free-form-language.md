---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2511.22963"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_commanding-humanoid-by-free-form-language.md
summary: "让人形听懂并执行自由形式自然语言指令，是迈向无缝人机交互与通用具身智能的关键一步。本文提出 Humanoid-LLA（Large Language Action Model），针对两大核心难题——成对语言-动作数据稀缺与物理不稳定——给出解法：通过学习一个统一的人-人形动作词表（unified human-humanoid motion vocabulary），把高层语言语义与物理可控的底层控制连接起来；并采用一个新颖的两阶段微调框架：先做有监督的动作思维链（motion Chain-of-Thought）学习，再用物理反馈引导的强化学习精修。借助跨本体（cross-embodiment）设计实现泛化。结果：对新语言指令有更好泛化、生成多样动作且保持高物理保真，并在仿真与真实跨本体实验中验证。"
---

# Commanding Humanoid by Free-form Language

**Commanding Humanoid by Free-form Language: A Large Language Action Model with Unified Motion Vocabulary** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

让人形听懂并执行自由形式自然语言指令，是迈向无缝人机交互与通用具身智能的关键一步。本文提出 Humanoid-LLA（Large Language Action Model），针对两大核心难题——成对语言-动作数据稀缺与物理不稳定——给出解法：通过学习一个统一的人-人形动作词表（unified human-humanoid motion vocabulary），把高层语言语义与物理可控的底层控制连接起来；并采用一个新颖的两阶段微调框架：先做有监督的动作思维链（motion Chain-of-Thought）学习，再用物理反馈引导的强化学习精修。借助跨本体（cross-embodiment）设计实现泛化。结果：对新语言指令有更好泛化、生成多样动作且保持高物理保真，并在仿真与真实跨本体实验中验证。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| LLA | Large Language Action Model，大语言-动作模型 |
| Motion Vocabulary | 动作词表，统一表示人/人形动作的离散单元 |
| Motion CoT | 动作思维链，对动作做分步推理 |
| Cross-Embodiment | 跨本体，泛化到不同机器人形态 |
| Free-form Language | 自由形式语言，非模板化的自然指令 |
| Physical Feedback RL | 以物理反馈为奖励信号的强化学习 |

## 为什么重要

- **离散动作词表是连接 LLM 与控制的关键接口**，与 UniAct 的 FSQ 码本异曲同工；
- **「监督 CoT + 物理反馈 RL」是语言-动作模型的稳健训练范式**：先学映射、再用物理拉回可执行；
- **跨本体设计**有助于把一套语言接口推广到不同人形；
- **与 FRoM-W1、ULTRA、SafeFlow、UniAct 同属语言/多模态驱动全身控制簇**，可横向对照生成器与稳定化手段。

## 解决什么问题

用自由语言指挥人形面临两道坎： - **数据稀缺**：高质量**成对的语言-动作**数据少； - **物理不稳定**：语言生成的动作未必物理可执行/稳定。

论文要：用一个**统一动作词表**把语言与物理控制对齐，并通过训练让模型对**新指令泛化**且**物理保真**。

## 核心机制

1. **统一人-人形动作词表**：连接语言语义与物理可控控制，缓解数据稀缺与模态鸿沟；
2. **两阶段微调**：有监督动作思维链 + 物理反馈 RL，兼顾映射准确与物理稳定；
3. **跨本体泛化**：可迁移不同本体；
4. **强泛化 + 高保真**：对新语言指令泛化，生成多样且物理可信的动作。

方法拆解（深读笔记小节）：统一人-人形动作词表；两阶段微调；跨本体设计；评测；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Commanding_Humanoid_by_Free-form_Language__LLA_with_Unified_Motion_Vocabulary/Commanding_Humanoid_by_Free-form_Language__LLA_with_Unified_Motion_Vocabulary.html> |
| arXiv | <https://arxiv.org/abs/2511.22963> |
| 作者 | Zhirui Liu、Kaiyang Ji、Ke Yang、Yahao Fan、Jingyi Yu、Ye Shi、Jingya Wang（上科大等） |
| 发表 | 2025 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_commanding-humanoid-by-free-form-language.md](../../sources/papers/humanoid_pnb_commanding-humanoid-by-free-form-language.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Commanding_Humanoid_by_Free-form_Language__LLA_with_Unified_Motion_Vocabulary/Commanding_Humanoid_by_Free-form_Language__LLA_with_Unified_Motion_Vocabulary.html>
- 论文：<https://arxiv.org/abs/2511.22963>

## 推荐继续阅读

- [机器人论文阅读笔记：Commanding Humanoid by Free-form Language](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Commanding_Humanoid_by_Free-form_Language__LLA_with_Unified_Motion_Vocabulary/Commanding_Humanoid_by_Free-form_Language__LLA_with_Unified_Motion_Vocabulary.html)
