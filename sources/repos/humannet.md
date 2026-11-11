# HumanNet（百万小时人中心视频语料）

- **标题**: HumanNet
- **论文**: https://arxiv.org/abs/2605.06747 — *HumanNet: Scaling Human-centric Video Learning to One Million Hours*
- **项目页**: https://dagroup-pku.github.io/HumanNet/
- **代码 / 发布**: https://github.com/DAGroup-PKU/HumanNet/
- **类型**: dataset / code-release / paper
- **机构**: 北京大学 DAGroup
- **收录日期**: 2026-05-14

## 一句话摘要

约 **一百万小时**、一三人称混合的 **人中心** 互联网级视频语料，配套交互导向标注与可扩展策展管线；面向表示学习、活动理解、运动生成与人–机迁移等具身下游，并在论文中给出基于 **LingBot-VLA** 的受控后训练对比（egocentric 子集 vs 真机小时）。

## 为何值得保留

- **规模与覆盖**：与 EPIC-KITCHENS、Ego4D、EgoScale、[EgoVerse](../../wiki/entities/paper-egoverse.md)、Ego-Exo4D 等并列讨论时，HumanNet 在「总时长 × 视点 × 活动广度」上给出新的数量级参照。
- **工程可拆分**：采集 / 处理 / 标注三阶段边界清晰，便于对照本仓库中 [自动化标注流水线](../../wiki/methods/auto-labeling-pipelines.md) 与 [具身数据清洗](../../wiki/concepts/embodied-data-cleaning.md) 等条目。
- **开源入口**：论文 + 项目页 + GitHub 并列，便于跟踪数据卡、许可与版本发布。

## 对 Wiki 的映射

- **wiki/entities/humannet.md**：数据集与管线实体页（归纳级，非论文转存）。
- **wiki/methods/vla.md**、**wiki/methods/imitation-learning.md**：补充「人类视频作为 VLA/IL 预训练数据来源」的当代参照。

## 2026-11-11：源码分支与数据集入口核对

- [HumanNet main](https://github.com/DAGroup-PKU/HumanNet/tree/main) 是原始百万小时**人中心（含一三人称）**数据论文与仓库入口；main 的完整数据集开放项目仍未标完成。
- [Open-HumanWAM 独立分支](https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM) 有 HumanWAM-Vanilla 训练/评测入口及 HumanWAM-Goal 推理接口；源码运行时序图在 [Open-HumanWAM 独立实体](../../wiki/entities/paper-open-humanwam.md)，**不是 HumanNet 全量数据的下载代码**。
- Open-HumanWAM 分支 README 列出 [HF 50K 语料](https://huggingface.co/datasets/DAGroup-PKU/Open-HumanWAM-50K) 与 [ModelScope 50K 镜像](https://modelscope.cn/datasets/DAGroup-PKU/EgoCorpus-50kH-LeRobot-v3)；这两处不要与[HumanNet ModelScope 原始 Ego 入口](../datasets/humannet-modelscope-ego.md)混淆。
