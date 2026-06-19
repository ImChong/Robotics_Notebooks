---
type: entity
tags: [dataset, bfm, behavior-foundation-model, human-motion, awesome-bfm-papers]
status: complete
updated: 2026-05-26
summary: "大规模人形动作数据入口，贴近 BFM goal-conditioned scaling。"
related:
  - ../concepts/behavior-foundation-model.md
  - ../overview/bfm-41-papers-technology-map.md
sources:
  - ../../sources/papers/bfm_awesome_dataset_humanoid_x_arxiv_2501_05098.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
---

# Humanoid-X（BFM 行为数据）

**Humanoid-X** 列入 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) 数据集表（2025 · Humanoids）。本页为 **索引级** 说明；规模与许可以官方页面为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |

## 为什么重要

- 大规模人形动作数据入口，贴近 BFM goal-conditioned scaling。
- BFM 数据链路的瓶颈往往在 **能否变成机器人可信、可执行、可迁移** 的训练材料，而非单纯 clip 数量（见 [BFM 技术地图](../overview/bfm-41-papers-technology-map.md) § 数据集）。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 规模（列表标注） | 163800 clips · 240.0 h |
| 论文/说明 | <https://arxiv.org/abs/2501.05098> |
| 入口 | <https://github.com/sihengz02/UH-1> |
| 许可证 | 研究用途；继承上游视频/动捕源许可，商用以官方页面为准 |
| 重定向就绪度 | 含人形动作表示，但跨具体本体仍需按目标形态重定向/适配后作策略输入 |

## 与其他页面的关系

- [behavior-foundation-model.md](../concepts/behavior-foundation-model.md)
- [bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)

## 参考来源

- [bfm_awesome_dataset_humanoid_x_arxiv_2501_05098.md](../../sources/papers/bfm_awesome_dataset_humanoid_x_arxiv_2501_05098.md)
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md#数据集10)
- 数据集页：<https://arxiv.org/abs/2501.05098>

## 推荐继续阅读

- [机器人论文阅读笔记：UH-1 Learning from Massive Human Videos for Universal Humanoid Pose Control](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/UH-1_Learning_from_Massive_Human_Videos_for_Universal_Humanoid_Pose_Control/UH-1_Learning_from_Massive_Human_Videos_for_Universal_Humanoid_Pose_Control.html)
- [awesome-bfm-papers § Datasets](https://github.com/friedrichyuan/awesome-bfm-papers#datasets)
