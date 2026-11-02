---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-o2o-afford-annotation-free-large-scale-object-object-aff
arxiv: "2106.15087"
venue: "CoRL 2021"
summary: "CoRL 2021 的 affordance 论文，清单分组：Perception（Scope: Object-pair；Perception Subcategory: Spatial）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_o2o-afford-annotation-free-large-scale-object-object-aff.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# O2O-Afford

**O2O-Afford: Annotation-Free Large-Scale Object-Object Affordance Learning**（CoRL 2021）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

O2O-Afford: Annotation-Free Large-Scale Object-Object Affordance Learning——在清单中归入 **Perception（Scope: Object-pair；Perception Subcategory: Spatial）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
    subgraph DATA["训练数据：仿真交互，无标注"]
        T["SAPIEN + ShapeNet / PartNet<br/>随机生成场景与 acting object<br/>四类任务：placement / fitting / pushing / stacking"]
        T --> TR["在 applicable 且 possible 的点 p 上<br/>执行硬编码短程直线轨迹 τ_T(p)"]
        TR --> OUT["按任务指标判定成功 / 失败<br/>作为逐点监督标签"]
    end
    subgraph NET["网络"]
        S["场景部分点云 S"] --> ES["Enc_scene（PointNet++）→ 场景逐点特征 F_S"]
        O["完整 acting object 点云 O"] --> EO["Enc_object（PointNet++）→ 物体逐点特征 F_O 与全局特征 F_g"]
        ES --> FPS["FPS 采样 k 个场景 seed 点"]
        FPS --> KER["Object-kernel 点卷积：把 O 移到每个 seed 点<br/>逆距离加权插值查询场景特征 F_(S|O)<br/>拼接 F_O 后经 PointNet（MLP + max-pool）得 seed 特征"]
        EO --> KER
        KER --> DEC["Dec_critic（MLP + Sigmoid）<br/>融合 seed 卷积特征、局部场景特征、全局物体特征"]
        EO --> DEC
        DEC --> INTERP["逆距离加权插值回全部场景点"]
        INTERP --> A["逐点 affordance 热力图 A_scene"]
    end
    OUT --> LOSS["二值交叉熵损失，端到端训练<br/>每个任务单独训练网络"]
    A --> LOSS
```

图依据：论文 §4 Task Definition and Data Generation（4.1–4.2）、§5 Method for Affordance Prediction / Fig.3（网络 §5.1，训练与损失 §5.2）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | CoRL 2021 |
| 方法名（清单标注） | O2O-Afford |
| 分组 1 | Perception · Scope: Object-pair；Perception Subcategory: Spatial |
| 论文 | <https://arxiv.org/abs/2106.15087> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；方法流程见上文「流程总览」，实验细节请读 [原文](https://arxiv.org/abs/2106.15087)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2106.15087) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「O2O-Afford」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_o2o-afford-annotation-free-large-scale-object-object-aff.md`](../../sources/papers/affordance_awesome_o2o-afford-annotation-free-large-scale-object-object-aff.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2106.15087>
- [arXiv PDF](https://arxiv.org/pdf/2106.15087)：已通读方法部分（实验与附录以略读为准）：§1 Introduction、§3 Problem Formulation、§4 Task Definition and Data Generation、§5 Method（5.1–5.2）、§6 实验设置（略读）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2106.15087)
