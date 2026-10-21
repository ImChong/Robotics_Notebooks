# Evo-DexBench 论文来源归档

## 元数据

- **题目：** Evo-DexBench: Benchmarking Generalization across Related Tasks for Dexterous Manipulation
- **类型：** 论文（cs.RO）
- **作者：** Bing Cheng、Yiran Mao、Zewei Ye、Yifan Jin、Xinyi Chen、Ruifeng Tu、Shuhuai Wang、Tao Lin、Zhimu Zhou、Yuxin Du、Fangwei Zhong、Junchi Yan、Bo Zhao
- **预印本：** https://arxiv.org/abs/2610.21180
- **版本 / 日期：** v1，2026-10-15
- **代码：** https://github.com/MINT-SJTU/Evo-DexBench（归档：[仓库来源](../repos/evo_dexbench.md)）
- **数据与资产：** [Hugging Face collection](https://huggingface.co/collections/MINT-SJTU/evo-dexbench)；仓库 README 列出 [数据集](https://huggingface.co/datasets/MINT-SJTU/Evo-DexBench-dataset) 与 [资产](https://huggingface.co/datasets/MINT-SJTU/Evo-DexBench_Assets)
- **项目页：** https://mint-sjtu.github.io/Evo-DexBench.io/（仓库 README 检查时标注为 coming soon；归档：[项目页核查](../sites/evo-dexbench-project-page.md)）

## 论文要解决的问题

灵巧操作策略需要把已有相关任务中学到的功能性交互迁移到新任务，而不是靠大量目标任务示范记忆动作。论文指出，既有 benchmark 的目标示范常常过多，且源任务与目标任务的相关性没有被控制，因此难以区分迁移能力与记忆。

## 核心设定与报告结果

- 提出 **Core–Gen** 原则：Core 是与目标共享功能性交互的相关源任务；Gen 是同一功能家族中的新目标任务，只提供少量示范。
- Evo-DexBench 包含 **30 个任务、6 个功能家族**，使用 VR 遥操作流程；每项任务有 3 个日常场景布局。
- 数据集含 **1,248 条多模态示范**，由带 **123 个触觉 taxel** 的模拟 Shadow Hand 采集。
- 摘要报告最佳方法平均成功率为 **38.5%**，说明相关任务间复用功能性交互仍有挑战。
- 在两个任务上，联合 Core+Gen 训练相较于先 Core 训练再对 Gen 微调，成功率分别提升 **12%** 与 **38%**；匹配功能的 Core 数据比不匹配数据收益更大。

## 阅读边界

以上数字与结论来自 arXiv v1 摘要。详细任务划分、训练设定、逐基线结果和指标定义应以论文正文及官方文档为准；不要把摘要中的提升百分比推广为所有任务的总体结论。

## Robotics_Notebooks 映射

- 项目实体：[Evo-DexBench](../../wiki/entities/paper-evo-dexbench.md)
- 仓库来源：[evo_dexbench](../repos/evo_dexbench.md)
- 数据与资产：[evo_dexbench_dataset](../datasets/evo_dexbench_dataset.md)
