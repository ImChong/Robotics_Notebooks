---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-strap-structured-object-affordance-segmentation-with-poi
arxiv: "2304.08492"
venue: "arXiv 2023"
summary: "arXiv 2023 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Visual）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_strap-structured-object-affordance-segmentation-with-poi.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# STRAP

**STRAP: Structured Object Affordance Segmentation with Point Supervision**（arXiv 2023）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

STRAP: Structured Object Affordance Segmentation with Point Supervision——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Visual）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    X["输入图像 X + 稀疏点标注 / 图像级 affordance 标签"] --> PL["点标注转伪标签 Y：以标注点为中心画半径 σ_d 的圆盘"]
    X --> BB["ResNetV2-50 + Transformer 编码器（readout token）；重组 4 层 token，经 RefineNet 融合块逐步上采样得特征 F"]
    BB --> DP["稠密预测头：全连接 + 双线性插值，输出 affordance 分割图"]
    BB --> LAI["LAI 头（Label Affinity Inference）：隐式物体层 + 显式 affordance 层，双向 RNN 建模 inter / intra 关系，输出图像级 affordance 标签"]
    DP --> LP["PCE 损失（对 Y）"]
    PL --> LP
    DP --> LS["空间损失 L_spatial：基于 RGBXY 核的稀疏连接 CRF（空间亲和）"]
    X --> LS
    LAI --> LL["标签损失 L_label（BCE）"]
    subgraph ST["三阶段训练"]
        S1["阶段一：PCE + 空间损失"] --> S2["阶段二：加入 LAI 头，PCE + 空间 + 标签损失"]
        S2 --> S3["阶段三：EM 迭代：E 步冻结网络预测并以 flood-fill 从点标注扩展出前景区域作伪标签；M 步用伪标签微调"]
    end
    LP --> S1
    LS --> S1
    LL --> S2
```

图依据：论文 §3.1–3.4（Fig.2、Fig.3、Fig.4）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2023 |
| 方法名（清单标注） | STRAP |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Visual |
| 论文 | <https://arxiv.org/abs/2304.08492> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要。方法流程见上方「流程总览」（依据论文方法部分）；需要机制与实验请读 [原文](https://arxiv.org/abs/2304.08492)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2304.08492) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「STRAP」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的只有标题、出处与分类标签；方法与结果未经本页核实。
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

- [`sources/papers/affordance_awesome_strap-structured-object-affordance-segmentation-with-poi.md`](../../sources/papers/affordance_awesome_strap-structured-object-affordance-segmentation-with-poi.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2304.08492>
- [arXiv PDF](https://arxiv.org/pdf/2304.08492)：已通读方法部分（实验与附录以略读为准）：§3 方法（3.1 网络结构、3.2 空间亲和、3.3 标签亲和、3.4 训练）、§4 实验设置

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2304.08492)
