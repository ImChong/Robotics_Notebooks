---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-afun-towards-an-affordance-foundation-model-for-function
arxiv: "2606.02551"
venue: "arXiv 2026"
summary: "arXiv 2026 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Visual · Spatial · Generalizable）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_afun-towards-an-affordance-foundation-model-for-function.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# AFUN

**AFUN: Towards an Affordance Foundation Model for Functionality Understanding**（arXiv 2026）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

AFUN: Towards an Affordance Foundation Model for Functionality Understanding——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Visual · Spatial · Generalizable）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    subgraph 数据["数据流水线（Fig.2）"]
        R["10 个公开来源视频<br/>人类 / 机器人 / 仿真 / 真实扫描"] --> PP["切分动作区间<br/>标准化 RGB-D、任务语言、相机参数"]
        PP --> TK["VLM 生成可操作部件 query<br/>SAM3 跟踪被操作物体"]
        TK --> OUT["功能 mask + 物体轨迹"]
        OUT --> BZ["反投影取 3D 均值<br/>拟合 Bézier 样条曲线"]
        BZ --> FL["逐步过滤 + 人工质检<br/>59,867 训练样本"]
    end
    subgraph 模型["AFUN 前向（Fig.4）"]
        IN["RGB-D + 任务短语"] --> VLM["冻结 Qwen3-VL<br/>附加 MetaQuery：mq_s 与 mq_m"]
        VLM -->|mq_s 经两层 MLP| SAM["SAM3 mask decoder<br/>功能 mask + per-object 特征"]
        IN --> ENC["Sonata 3D 特征编码器<br/>点云 → geo feat"]
        ENC --> MD["Motion Decoder<br/>transformer decoder"]
        SAM -->|per-object 特征| MD
        VLM -->|mq_m| MD
        MD --> CV["3D Bézier 控制点<br/>P0 为 mask 深度质心"]
        SAM --> DEP["mask 反投影为 3D 点<br/>交给 AnyGrasp 等抓取模块"]
    end
        subgraph 训练["三阶段训练"]
        S1["Stage 1：MetaQuery–SAM3 对齐<br/>Visual Genome，MSE"] --> S2["Stage 2：端到端分割训练<br/>四个 affordance 数据集"] --> S3["Stage 3：分割 + 运动联合训练<br/>L_sam3 + L_curve"]
    end
    FL -.->|Stage 3 训练数据| S3
```

图依据：论文 §3 数据流水线（Fig.2）、§4.1 Network Architecture（Fig.4）、§4.2 Training Scheme；引言说明 mask 可反投影后交给 AnyGrasp 等抓取模块。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | arXiv 2026 |
| 方法名（清单标注） | AFUN |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Visual · Spatial · Generalizable |
| 论文 | <https://arxiv.org/abs/2606.02551> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，没有方法摘要；网络结构与流程见上方「流程总览」（依据论文方法部分），训练细节与实验请读 [原文](https://arxiv.org/abs/2606.02551)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2606.02551) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「AFUN」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_afun-towards-an-affordance-foundation-model-for-function.md`](../../sources/papers/affordance_awesome_afun-towards-an-affordance-foundation-model-for-function.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2606.02551>
- [arXiv PDF](https://arxiv.org/pdf/2606.02551)：已通读方法部分（实验与附录以略读为准）：引言数据流水线（Dataset Curation 至 Filtering）、§4.1–§4.2

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2606.02551)
