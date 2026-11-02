---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-closed-loop-transfer-for-weakly-supervised-affordance-gr
arxiv: "2510.17384"
venue: "ICCV 2025"
summary: "ICCV 2025 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Visual）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_closed-loop-transfer-for-weakly-supervised-affordance-gr.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# LoopTrans

**Closed-Loop Transfer for Weakly-supervised Affordance Grounding**（ICCV 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Closed-Loop Transfer for Weakly-supervised Affordance Grounding——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Visual）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
flowchart TB
    IN["输入：成对的 exocentric / egocentric 图像<br/>+ 图像级 affordance 类别标签"]
    F["DINO 提取 patch 特征 F_exo / F_ego"]
    subgraph S1["阶段 1：Interaction → Activation"]
        SC["Shared CAM（SCAM）<br/>exo 与 ego 共享参数联合训练<br/>得到激活图 G_exo / G_ego"]
        NH["noise-absorbing heads<br/>吸收 exo 图中的背景与人体噪声"]
    end
    subgraph S2["阶段 2：Activation → Localization"]
        CL["DINO 特征无监督聚类<br/>把 ego 图划分为 K 个物体部件"]
        PM["选出与 G_ego 阈值化区域 IoU 最大的部件<br/>作为伪 mask"]
        PD["Pixel Decoder 以伪 mask 监督<br/>输出 ego 逐像素 affordance 定位图"]
        CL --> PM --> PD
    end
    subgraph S3["阶段 3：Localization → Activation"]
        DD["Denoising Distillation<br/>exo 激活特征向 ego 定位特征靠拢<br/>并远离噪声特征"]
    end
    IN --> F --> SC
    SC --- NH
    SC -->|G_ego| CL
    F -->|F_ego| CL
    F -->|F_ego| PD
    PD --> OUT["输出：egocentric affordance 定位图"]
    PD -->|定位特征反馈| DD
    NH --> DD
    DD -->|反向蒸馏回 SCAM| SC
```

图依据：论文 §3 Preliminary（特征提取与 CAM）、§4 Method（§4.1 Unified Exo-to-Ego Activation、§4.2 Region Activation to Pixel Localization、§4.3 Ego-to-Exo Denoising Distillation，Fig. 3 / Fig. 4）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ICCV 2025 |
| 方法名（清单标注） | LoopTrans |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Visual |
| 论文 | <https://arxiv.org/abs/2510.17384> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；方法流程见上文「流程总览」（依据论文方法部分），实验与结果请读 [原文](https://arxiv.org/abs/2510.17384)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2510.17384) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「LoopTrans」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_closed-loop-transfer-for-weakly-supervised-affordance-gr.md`](../../sources/papers/affordance_awesome_closed-loop-transfer-for-weakly-supervised-affordance-gr.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2510.17384>
- [arXiv PDF](https://arxiv.org/pdf/2510.17384)：已通读方法部分（实验与附录以略读为准）：§3 Preliminary、§4 Method（§4.1–4.3）、§5.1 数据集与实现

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2510.17384)
