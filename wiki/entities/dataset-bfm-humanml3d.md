---

type: entity
tags: [dataset, bfm, behavior-foundation-model, human-motion, awesome-bfm-papers, pku]
status: complete
updated: 2026-05-26
summary: "文本到人体动作生成常用基准。"
related:
  - ../concepts/behavior-foundation-model.md
  - ../overview/bfm-41-papers-technology-map.md
  - ../entities/amass.md
sources:
  - ../../sources/papers/bfm_awesome_dataset_humanml3d_cvpr_2022.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
---

# HumanML3D（BFM 行为数据）

**HumanML3D** 列入 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) 数据集表（2022 · CVPR）。本页为 **索引级** 说明；规模与许可以官方页面为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |

## 为什么重要

- 文本到人体动作生成常用基准。
- BFM 数据链路的瓶颈往往在 **能否变成机器人可信、可执行、可迁移** 的训练材料，而非单纯 clip 数量（见 [BFM 技术地图](../overview/bfm-41-papers-technology-map.md) § 数据集）。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 规模（列表标注） | 14616 clips · 28.6 h |
| 论文/说明 | <https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.pdf> |
| 入口 | <https://github.com/EricGuo5513/HumanML3D> |
| 许可证 | 代码 MIT；动作源自 AMASS + HumanAct12，学术研究用途；继承上游源数据（如 AMASS）许可，商用授权以官方页面为准 |
| 重定向就绪度 | 人体骨架/SMPL 原生，喂人形机器人前需 **retarget 重定向** 到目标形态，非即插即用策略输入 |

## 与其他页面的关系

- [behavior-foundation-model.md](../concepts/behavior-foundation-model.md)
- [bfm-41-papers-technology-map.md](../overview/bfm-41-papers-technology-map.md)

## 参考来源

- [bfm_awesome_dataset_humanml3d_cvpr_2022.md](../../sources/papers/bfm_awesome_dataset_humanml3d_cvpr_2022.md)
- [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md#数据集10)
- 数据集页：<https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.pdf>

## 推荐继续阅读

- [awesome-bfm-papers § Datasets](https://github.com/friedrichyuan/awesome-bfm-papers#datasets)
