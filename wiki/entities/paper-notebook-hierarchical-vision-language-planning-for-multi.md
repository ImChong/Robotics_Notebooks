---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2506.22827"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_hierarchical-vision-language-planning-for-multi.md
summary: "让人形可靠执行复杂多步操作对工业/家庭部署很关键。本文提出一个分层规划与控制框架，含三层：① 底层——基于 RL 的控制器，负责跟踪全身动作目标；② 中层——一组用模仿学习训练的技能策略，为任务各步产生动作目标；③ 高层——一个视觉-语言规划模块，用预训练 VLM 决定执行哪个技能并实时监控其完成。在 Unitree G1 人形上做非抓握式（non-prehensile）取放任务、40+ 次真机试验，完整序列成功率 73%。"
---

# Hierarchical Vision-Language Planning for Multi-Step Humanoid Manipulation

**Hierarchical Vision-Language Planning for Multi-Step Humanoid Manipulation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

让人形可靠执行复杂多步操作对工业/家庭部署很关键。本文提出一个分层规划与控制框架，含三层：① 底层——基于 RL 的控制器，负责跟踪全身动作目标；② 中层——一组用模仿学习训练的技能策略，为任务各步产生动作目标；③ 高层——一个视觉-语言规划模块，用预训练 VLM 决定执行哪个技能并实时监控其完成。在 Unitree G1 人形上做非抓握式（non-prehensile）取放任务、40+ 次真机试验，完整序列成功率 73%。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Hierarchical | 分层（高/中/低三层） |
| VLM | Vision-Language Model |
| Skill Policy | 技能策略（中层，IL 训练） |
| Whole-Body Tracking | 全身动作目标跟踪（底层 RL） |
| Non-prehensile | 非抓握式（推/拨等） |
| Real-time Monitoring | 实时监控技能完成 |

## 为什么重要

- **分层是多步长序列任务的可靠之道**：高层决策、中层技能、底层控制各司其职；
- **VLM 实时监控"某步是否完成"**是闭环关键，避免盲目推进；
- **非抓握操作**（推/拨）拓展了人形操作类型；
- 与 Proprio-MLLM、BiBo 等 VLM 规划工作互补。

## 解决什么问题

人形**多步操作**要可靠： - 单层端到端难覆盖**长序列**； - 需要**高层决策 + 中层技能 + 底层控制**协同； - 还要**实时知道某步是否完成**。

论文要：一个**分层、可监控**的多步人形操作框架。

## 核心机制

1. **三层规划-控制框架**：VLM 规划 + IL 技能 + RL 全身控制；
2. **VLM 高层选技能 + 实时监控完成**；
3. **多步可靠执行**：面向工业/家庭长序列任务；
4. **真机验证**：G1 非抓握取放，序列成功率 73%。

方法拆解（深读笔记小节）：三层架构；VLM 高层：选技能 + 监控；评测；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Hierarchical_Vision-Language_Planning_for_Multi-Step_Humanoid_Manipulation/Hierarchical_Vision-Language_Planning_for_Multi-Step_Humanoid_Manipulation.html> |
| arXiv | <https://arxiv.org/abs/2506.22827> |
| 作者 | André Schakkal、Ben Zandonati、Zhutian Yang、Navid Azizan（MIT） |
| 发表 | 2025 年 6 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_hierarchical-vision-language-planning-for-multi.md](../../sources/papers/humanoid_pnb_hierarchical-vision-language-planning-for-multi.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Hierarchical_Vision-Language_Planning_for_Multi-Step_Humanoid_Manipulation/Hierarchical_Vision-Language_Planning_for_Multi-Step_Humanoid_Manipulation.html>
- 论文：<https://arxiv.org/abs/2506.22827>

## 推荐继续阅读

- [机器人论文阅读笔记：Hierarchical Vision-Language Planning for Multi-Step Humanoid Manipulation](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Hierarchical_Vision-Language_Planning_for_Multi-Step_Humanoid_Manipulation/Hierarchical_Vision-Language_Planning_for_Multi-Step_Humanoid_Manipulation.html)
