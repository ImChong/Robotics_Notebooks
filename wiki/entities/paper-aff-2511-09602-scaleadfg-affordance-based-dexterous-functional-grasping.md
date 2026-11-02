---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-scaleadfg-affordance-based-dexterous-functional-grasping
arxiv: "2511.09602"
venue: "RA-L 2025"
summary: "RA-L 2025 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action） ｜ Datasets / Benchmarks（Data Modality: 3D/grasp）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_scaleadfg-affordance-based-dexterous-functional-grasping.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# ScaleADFG

**ScaleADFG: Affordance-based Dexterous Functional Grasping via Scalable Dataset**（RA-L 2025）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

ScaleADFG: Affordance-based Dexterous Functional Grasping via Scalable Dataset——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action） ｜ Datasets / Benchmarks（Data Modality: 3D/grasp）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
    subgraph DS["ScaleADFG-Dataset 自动构建"]
        A["互联网搜索图像"] --> B["Grounded SAM 2 过滤与分割"]
        B --> C["预训练生成模型生成 3D 资产<br/>InstantMesh / TripoSR / TRELLIS（随机选一）"]
        C --> D["Affordance Retrieval<br/>PTv3-object 提取点云特征<br/>与类别模板做稠密特征匹配得到功能部位与抓握部位"]
        D --> E["Axis Alignment Initialization<br/>对齐手与物体的 GF 轴、FA 轴"]
        E --> F["Affordance-based Optimization<br/>功能部位距离 + 抓握部位距离 + 力闭合 + 手物穿透 + 手自穿透"]
        F --> G["质量指标过滤 + 仿真精修筛选"]
        G --> H["功能性灵巧抓取数据集"]
    end
    subgraph NET["ScaleADFG-Net（CVAE）"]
        I["物体点云 + PTv3-object 特征的前三个主成分拼接<br/>（网络整合 PTv3-object 与 PointNet）"]
        J["Encoder → 隐变量 → Decoder<br/>预测抓取配置"]
        K["可微前向运动学 FK"]
        I --> J --> K
    end
    H --> |训练：重建损失 + KL 散度| J
    K --> L["推理：去掉 Encoder，从 N(0,I) 采样隐变量，输出抓取配置"]
```

图依据：论文 §III（数据集合成：A 3D 资产生成、B Affordance Retrieval、C 抓取优化）、§IV（ScaleADFG-Net）、Fig. 2

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | RA-L 2025 |
| 方法名（清单标注） | ScaleADFG |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 分组 2 | Datasets / Benchmarks · Data Modality: 3D/grasp |
| 论文 | <https://arxiv.org/abs/2511.09602> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；方法流程见上文「流程总览」（依据论文方法部分），实验与结果请读 [原文](https://arxiv.org/abs/2511.09602)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2511.09602) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「ScaleADFG」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

- 可确证的有标题、出处与分类标签，以及「流程总览」所述方法流程（依据论文方法部分）；实验结果未经本页核实。
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

- [`sources/papers/affordance_awesome_scaleadfg-affordance-based-dexterous-functional-grasping.md`](../../sources/papers/affordance_awesome_scaleadfg-affordance-based-dexterous-functional-grasping.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2511.09602>
- [arXiv PDF](https://arxiv.org/pdf/2511.09602)：已通读方法部分（实验与附录以略读为准）：§III 数据集合成、§IV ScaleADFG-Net、§V-A 评测指标

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2511.09602)
