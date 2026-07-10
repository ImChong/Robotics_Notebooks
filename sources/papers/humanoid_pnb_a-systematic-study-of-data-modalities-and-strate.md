# A Systematic Study of Data Modalities and Strategies for Co-training Large Behavior Models for Robot Manipulation

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** A Systematic Study of Data Modalities and Strategies for Co-training Large Behavior Models for Robot Manipulation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models.html>
- **分类：** 06_Manipulation
- **arXiv：** <https://arxiv.org/abs/2602.01067>
- **入库日期：** 2026-07-10
- **一句话说明：** 大行为模型（Large Behavior Models）把模仿学习扩展到多任务机器人数据的大规模训练，展现强灵巧操作能力，但泛化仍受限于机器人数据覆盖不足。为在不昂贵额外采集的前提下扩覆盖，近期工作依赖协同训练（co-training）：联合学习目标机器人数据与异构数据模态。但不同协同训练数据模态与策略如何影响策略性能仍理解不足。本文做系统研究：在 4000 小时机器人/人类操作数据 + 5000 万视觉-语言样本上，跨 89 个策略、5.8 万次仿真 + 2835 次真机 rollout，比较五类模态（视觉-语言数据、稠密语言标注、跨本体机器人数据、人类视频、离散动作 token）与单/多阶段训练策略。主要发现：视觉-语言与跨本体机器人数据显著提升对分布偏移、新任务与语言理解的泛化；离散动作 token 收益甚微；组合有效模态可累加增益；纯机器人训练会损害视觉-语言能力，而协同训练能恢复；思维链（CoT）条件在其基准上无性能增益。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-a-systematic-study-of-data-modalities-and-strate](../../wiki/entities/paper-notebook-a-systematic-study-of-data-modalities-and-strate.md).

## 对 wiki 的映射

- [paper-notebook-a-systematic-study-of-data-modalities-and-strate](../../wiki/entities/paper-notebook-a-systematic-study-of-data-modalities-and-strate.md)
- 分类父节点：[paper-notebook-category-06-manipulation](../../wiki/overview/paper-notebook-category-06-manipulation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models/A_Systematic_Study_of_Data_Modalities_and_Strategies_for_Co-training_Behavior_Models.html>
- 论文：<https://arxiv.org/abs/2602.01067>
