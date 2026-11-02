---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-locate-localize-and-transfer-object-parts-for-weakly-sup
arxiv: "2303.09665"
venue: "CVPR 2023"
summary: "CVPR 2023 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Visual · Interaction-driven）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_locate-localize-and-transfer-object-parts-for-weakly-sup.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# LOCATE

**LOCATE: Localize and Transfer Object Parts for Weakly Supervised Affordance Grounding**（CVPR 2023）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

LOCATE: Localize and Transfer Object Parts for Weakly Supervised Affordance Grounding——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Visual · Interaction-driven）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
    EXO["输入：多张 exocentric 交互图像"] --> DINO["DINO-ViT 提取深层特征 Fexo"]
    EGO["输入：一张 egocentric 物体图像"] --> DINO2["DINO-ViT 提取深层特征 Fego"]
    DINO --> CAM["类别感知卷积 ψcam 生成定位图 Pexo，GAP 得分类分数，Lcls"]
    CAM --> EMB["按阈值 τ 取高激活位置的 embedding，拼接为 fexo"]
    EMB --> PS["PartSelect：k-means 得 K 个 prototype"]
    DINO2 --> PS
    PS --> SIM["prototype 与 Fego 余弦距离得相似图，结合 DINO 自注意力显著图计算 PartIoU"]
    SIM --> SEL["PartIoU 最大且高于阈值 µ 的 prototype 作为物体部件表示 fop"]
    DINO2 --> CAME["ego 分支 ψcam 生成定位图 Pego，Lcls"]
    CAME --> MAP["与特征做 masked average pooling 得 ego embedding"]
    SEL --> COS["余弦嵌入损失 Lcos 监督 ego embedding 向 fop 靠拢"]
    MAP --> COS
    CAME --> CON["集中损失 Lc 使 Pego 激活集中连通"]
    CAME --> OUT["测试时仅保留 ego 分支：按 affordance 标签取第 t 张定位图作预测"]
```

图依据：论文 §3 / Fig.3、Fig.4

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CVPR 2023 |
| 方法名（清单标注） | LOCATE |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Visual · Interaction-driven |
| 论文 | <https://arxiv.org/abs/2303.09665> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；方法流程见上方「流程总览」（依据论文方法部分），实验细节请读 [原文](https://arxiv.org/abs/2303.09665)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2303.09665) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「LOCATE」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
- 开源状态清单未标注，入库时未打开项目页核查，复现前请自行确认。
- 要深读：从原文入手，再回综述实体页对照 perception → reasoning → action 的位置。

## 常见误区

1. 不要把分类标签当作论文结论；标签来自清单维护者的归类。
2. 不要把本页当作方法证明；它只是清单入口。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[vla.md](../methods/vla.md)

## 参考来源

- [`sources/papers/affordance_awesome_locate-localize-and-transfer-object-parts-for-weakly-sup.md`](../../sources/papers/affordance_awesome_locate-localize-and-transfer-object-parts-for-weakly-sup.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2303.09665>
- [arXiv PDF](https://arxiv.org/pdf/2303.09665)：已通读方法部分（实验与附录以略读为准）：§3 Method（3.1 定位交互区域、3.2 PartSelect、3.3 部件级知识迁移与总损失）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2303.09665)
