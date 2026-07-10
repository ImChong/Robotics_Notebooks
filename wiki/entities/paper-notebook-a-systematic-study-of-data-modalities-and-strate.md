---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2602.01067"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_a-systematic-study-of-data-modalities-and-strate.md
summary: "大行为模型（Large Behavior Models）把模仿学习扩展到多任务机器人数据的大规模训练，展现强灵巧操作能力，但泛化仍受限于机器人数据覆盖不足。为在不昂贵额外采集的前提下扩覆盖，近期工作依赖协同训练（co-training）：联合学习目标机器人数据与异构数据模态。但不同协同训练数据模态与策略如何影响策略性能仍理解不足。本文做系统研究：在 4000 小时机器人/人类操作数据 + 5000 万视觉-语言样本上，跨 89 个策略、5.8 万次仿真 + 2835 次真机 rollout，比较五类模态（视觉-语言数据、稠密语言标注、跨本体机器人数据、人类视频、离散动作 token）与单/多阶段训练策略。主要发现：视觉-语言与跨本体机器人数据显著提升对分布偏移、新任务与语言理解的泛化；离散动作 token 收益甚微；组合有效模态可累加增益；纯机器人训练会损害视觉-语言能力，而协同训练能恢复；思维链（CoT）条件在其基准上无性能增益。"
---

# A Systematic Study of Data Modalities and Strategies for Co-training Large Behavior Models for Robot Manipulation

**A Systematic Study of Data Modalities and Strategies for Co-training Large Behavior Models for Robot Manipulation** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

大行为模型（Large Behavior Models）把模仿学习扩展到多任务机器人数据的大规模训练，展现强灵巧操作能力，但泛化仍受限于机器人数据覆盖不足。为在不昂贵额外采集的前提下扩覆盖，近期工作依赖协同训练（co-training）：联合学习目标机器人数据与异构数据模态。但不同协同训练数据模态与策略如何影响策略性能仍理解不足。本文做系统研究：在 4000 小时机器人/人类操作数据 + 5000 万视觉-语言样本上，跨 89 个策略、5.8 万次仿真 + 2835 次真机 rollout，比较五类模态（视觉-语言数据、稠密语言标注、跨本体机器人数据、人类视频、离散动作 token）与单/多阶段训练策略。主要发现：视觉-语言与跨本体机器人数据显著提升对分布偏移、新任务与语言理解的泛化；离散动作 token 收益甚微；组合有效模态可累加增益；纯机器人训练会损害视觉-语言能力，而协同训练能恢复；思维链（CoT）条件在其基准上无性能增益。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| LBM | Large Behavior Model，大行为模型 |
| Co-training | 协同训练，目标数据 + 异构模态联合学 |
| Cross-Embodiment | 跨本体机器人数据 |
| Action Token | 离散动作 token |
| VLA | Vision-Language-Action |
| CoT | Chain-of-Thought，思维链条件 |

## 为什么重要

- **"加什么数据"比"加更多数据"更重要**：视觉-语言与跨本体最划算；
- **纯机器人训练会损害通用视觉-语言能力**——协同训练是必要的"保养"；
- **离散动作 token / CoT 未必有用**，提醒不要盲目堆技巧；
- 对人形 VLA/大行为模型的数据配方有直接指导（TRI 大规模实证）。

## 解决什么问题

大行为模型泛化受**机器人数据覆盖不足**所限： - 协同训练（加异构数据）能扩覆盖； - 但**哪些模态、哪种策略**有效**理解不足**，缺系统证据。

论文要：用**大规模、系统化**实验，厘清**协同训练**的数据模态与策略选择。

## 核心机制

1. **协同训练的系统性研究**：五模态 × 单/多阶段策略，大规模评测；
2. **关键结论**：视觉-语言 + 跨本体数据最有效，离散动作 token 收益小；
3. **组合累加 + 协同恢复**：有效模态可叠加，协同训练修复纯机器人训练的视觉-语言退化；
4. **CoT 无增益**：在其基准上的反直觉发现。

方法拆解（深读笔记小节）：五类协同训练数据模态；单/多阶段训练策略 + VLA 架构；大规模评测；主要发现；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models.html> |
| arXiv | <https://arxiv.org/abs/2602.01067> |
| 作者 | Fanqi Lin、Kushal Arora、Jean Mercat、Haruki Nishimura、Paarth Shah、Jose Barreiros 等（Toyota Research Institute, TRI） |
| 发表 | 2026 年 2 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_a-systematic-study-of-data-modalities-and-strate.md](../../sources/papers/humanoid_pnb_a-systematic-study-of-data-modalities-and-strate.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models.html>
- 论文：<https://arxiv.org/abs/2602.01067>

## 推荐继续阅读

- [机器人论文阅读笔记：A Systematic Study of Data Modalities and Strategies for Co-training Large Behavior Models for Robot Manipulation](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models.html)
