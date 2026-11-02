---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-where2explore-few-shot-affordance-learning-for-unseen-no
arxiv: "2309.07473"
venue: "NeurIPS 2023"
summary: "NeurIPS 2023 的 affordance 论文，清单分组：Perception（Scope: Object；Perception Subcategory: Spatial · Generalizable）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_where2explore-few-shot-affordance-learning-for-unseen-no.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Where2Explore

**Where2Explore: Few-shot Affordance Learning for Unseen Novel Categories of Articulated Objects**（NeurIPS 2023）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Where2Explore: Few-shot Affordance Learning for Unseen Novel Categories of Articulated Objects——在清单中归入 **Perception（Scope: Object；Perception Subcategory: Spatial · Generalizable）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
    subgraph 训练阶段
        A["训练类别划分：affordance 类别 C_aff 与相似性类别 C_sim"] --> B["在 C_aff 上学习逐点 affordance，形成 supporting set"]
        A --> C["相似性模块：对 C_sim 对象预测逐点相似度，监督信号 = affordance 预测对真值交互的准确率"]
        B -. 共享 PointNet++ 编码器 .- C
    end
    subgraph 新类别少样本探索循环
        D["新类别物体的部分点云"] --> E["相似性模块预测逐点相似度，并采样多种动作方向/夹爪朝向"]
        E --> F["选相似度最低的动作，执行短程轨迹，观察部件运动 m_i"]
        F --> G["用交互 (O_i, p_i, R_i, m_i) 更新 affordance 模块与相似性模块"]
        G --> H{"相似度达到阈值或交互预算用完？"}
        H -- 否 --> E
    end
    B --> D
    C --> D
    H -- 是 --> I["适配后的 affordance 可操作该类别未见物体"]
```

图依据：论文 §4（4.1–4.4）/ Fig.2、Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | NeurIPS 2023 |
| 方法名（清单标注） | Where2Explore |
| 分组 1 | Perception · Scope: Object；Perception Subcategory: Spatial · Generalizable |
| 论文 | <https://arxiv.org/abs/2309.07473> |

## 核心机制（归纳）

清单只给出标题、出处与分类标签，**没有**方法摘要。方法流程见上方「流程总览」（依据论文方法部分）；需要机制与实验请读 [原文](https://arxiv.org/abs/2309.07473)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2309.07473) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「Where2Explore」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_where2explore-few-shot-affordance-learning-for-unseen-no.md`](../../sources/papers/affordance_awesome_where2explore-few-shot-affordance-learning-for-unseen-no.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2309.07473>
- [arXiv PDF](https://arxiv.org/pdf/2309.07473)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3 问题定义、§4 方法（4.1–4.4）、§5.1–5.2 实验设定与基线

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2309.07473)
