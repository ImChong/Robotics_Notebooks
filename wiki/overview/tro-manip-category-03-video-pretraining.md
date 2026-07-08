---
type: overview
tags: [manipulation, category-hub, survey, video-pretraining, cross-embodiment]
status: complete
updated: 2026-07-08
summary: "T-RO 5 篇精选 · 03 无标签视频预训练（1 篇）— 人类操作视频如何经图到图生成变成可迁移操作策略？"
related:
  - ./tro-manip-5-papers-technology-map.md
  - ./tro-manip-category-02-representation.md
  - ./tro-manip-category-04-generative-models-survey.md
  - ../entities/paper-tro-manip-04-g3m.md
  - ../methods/mimic-video.md
sources:
  - ../../sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md
  - ../../sources/papers/tro_manip_5_papers_catalog.md
---

# T-RO 分类 03：无标签视频预训练

> **图谱分类节点**：对应 T-RO 2026 操作学习精选的 **03 无标签视频预训练** 分组（1 篇）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| G3M | Graph-to-Graphs Generative Modeling | 图到图生成建模（G3M 框架简称） |
| IL | Imitation Learning | 模仿学习 |
| LfD | Learning from Demonstrations | 从演示中学习 |

## 核心问题

**无动作标签的人类视频如何进入机器人策略？** 关键是用 **结构化图** 捕捉物体状态与手-物交互的空间关系，而非直接拟合像素或本体特定关节轨迹。

## 本组论文（1 篇）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 04 | G3M | [paper-tro-manip-04-g3m.md](../entities/paper-tro-manip-04-g3m.md) | [source](../../sources/papers/tro_manip_survey_04_g3m.md) |

## 与 GraphMimic 的分工

| 版本 | 出处 | wiki 节点 |
|------|------|-----------|
| **G3M** | IEEE T-RO 2026（期刊扩展） | [paper-tro-manip-04-g3m](../entities/paper-tro-manip-04-g3m.md) |
| GraphMimic | CVPR 2025 | 在 G3M 实体页交叉引用，不单独建重复节点 |

## 关联页面

- [T-RO 5 篇技术地图](./tro-manip-5-papers-technology-map.md)
- [mimic-video](../methods/mimic-video.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [wechat_shenlan_tro_manip_5_papers_survey.md](../../sources/blogs/wechat_shenlan_tro_manip_5_papers_survey.md)
- [tro_manip_5_papers_catalog.md](../../sources/papers/tro_manip_5_papers_catalog.md)

## 推荐继续阅读

- [Ego 9 篇技术地图](./ego-9-papers-technology-map.md)
