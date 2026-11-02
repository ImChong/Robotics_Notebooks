---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-language-embedded-radiance-fields-for-zero-shot-task-ori
arxiv: "2309.07970"
venue: "CoRL 2023"
summary: "CoRL 2023 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_language-embedded-radiance-fields-for-zero-shot-task-ori.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# LERF-TOGO

**Language Embedded Radiance Fields for Zero-Shot Task-Oriented Grasping**（CoRL 2023）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Language Embedded Radiance Fields for Zero-Shot Task-Oriented Grasping——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要，此处不替原文概括贡献）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Affordance | Action possibility (Gibson) | 环境/物体对 agent 提供的可交互机会 |
| HOI | Human-Object Interaction | 人–物交互，常作 affordance 监督来源 |
| VLM | Vision-Language Model | 视觉–语言模型，用于 affordance 推理与接地 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |

## 为什么重要

- 综述把 affordance 学习拆成感知、推理、引导动作三段（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），本条目是其中一段的代表文献之一。
- 可顺着 [技术地图](../overview/affordance-awesome-technology-map.md) 的同分组条目横向阅读同方向工作。

## 流程总览

```mermaid
flowchart TD
    A["腕载相机半球轨迹采集约 60 张图像（丢弃模糊帧）"] --> B["预提取 DINO 特征、多尺度 CLIP、ZoeDepth"]
    B --> C["训练 LERF：DINO 场 + 尺度条件 CLIP 场"]
    Q["自然语言查询：物体查询 + 部件查询"] --> D
    C --> D["俯视渲染，DINO 第一主成分阈值得前景掩码；物体查询找到最相关 3D 点"]
    D --> E["围绕该点渲染物体中心点云；按 DINO 相似度迭代 flood-fill，得 3D 物体掩码"]
    E --> F["条件 LERF 查询：仅在 3D 物体掩码内查询部件，得 3D 部件相关度"]
    C --> G["半球虚拟相机 + 预训练 GraspNet 采样 6-DoF 抓取，NMS 去重"]
    G --> H["抓取排序：s = 0.95 s_sem + 0.05 s_geom"]
    F --> H
    H --> I["执行排名最高的抓取并提起"]
    L["可选：LLM 少样本提示 + 多数投票生成物体/部件对与动作原语"] -.-> Q
```

图依据：论文 §4（4.1–4.4）/ Fig.3，LLM 部分见 §5

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CoRL 2023 |
| 方法名（清单标注） | LERF-TOGO |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2309.07970> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要。方法流程见上方「流程总览」（依据论文方法部分）；需要机制与实验请读 [原文](https://arxiv.org/abs/2309.07970)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2309.07970) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「LERF-TOGO」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[imitation-learning.md](../methods/imitation-learning.md)

## 参考来源

- [`sources/papers/affordance_awesome_language-embedded-radiance-fields-for-zero-shot-task-ori.md`](../../sources/papers/affordance_awesome_language-embedded-radiance-fields-for-zero-shot-task-ori.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2309.07970>
- [arXiv PDF](https://arxiv.org/pdf/2309.07970)：已通读方法部分（实验与附录以略读为准）：§2.3 LERF 预备、§3 问题设定、§4 方法（4.1–4.4）、§5 实验设定与 LLM 集成

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2309.07970)
