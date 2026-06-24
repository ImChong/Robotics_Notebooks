---

type: entity
tags: [dataset, bfm, behavior-foundation-model, human-motion, awesome-bfm-papers, inria]
status: complete
updated: 2026-05-26
summary: "语言描述与姿态桥接。"
related:
  - ../concepts/behavior-foundation-model.md
  - ../overview/bfm-41-papers-technology-map.md
sources:
  - ../../sources/papers/bfm_awesome_dataset_posescript_eccv_2022.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
---

# PoseScript（BFM 行为数据）

**PoseScript** 列入 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) 数据集表（2022 · ECCV）。本页为 **索引级** 说明；规模与许可以官方页面为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |

## 为什么重要

- 语言描述与姿态桥接。
- BFM 数据链路的瓶颈往往在 **能否变成机器人可信、可执行、可迁移** 的训练材料，而非单纯 clip 数量（见 [BFM 技术地图](../overview/bfm-41-papers-technology-map.md) § 数据集）。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 规模（列表标注） | - clips · - h |
| 论文/说明 | <https://arxiv.org/abs/2210.11795> |
| 入口 | <https://github.com/naver/posescript> |
| 模态 | SMPL 静态姿态（源自 AMASS）+ 自然语言 pose 描述 |
| 许可证 | NAVER 发布，研究用途；商用以官方页面为准 |
| 重定向就绪度 | SMPL 人体姿态，迁移到人形机器人需重定向到目标形态 |

## 与其他页面的关系

- [behavior-foundation-model.md](../concepts/behavior-foundation-model.md)
- [bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)

## 参考来源

- [bfm_awesome_dataset_posescript_eccv_2022.md](../../sources/papers/bfm_awesome_dataset_posescript_eccv_2022.md)
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md#数据集10)
- 数据集页：<https://arxiv.org/abs/2210.11795>

## 推荐继续阅读

- [awesome-bfm-papers § Datasets](https://github.com/friedrichyuan/awesome-bfm-papers#datasets)
