---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-vat-mart-learning-visual-action-trajectory-proposals-for
arxiv: "2106.14440"
venue: "ICLR 2022"
summary: "ICLR 2022 的 affordance 论文，清单分组：Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_vat-mart-learning-visual-action-trajectory-proposals-for.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# VAT-Mart

**VAT-Mart: Learning Visual Action Trajectory Proposals for Manipulating 3D Articulated Objects**（ICLR 2022）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

VAT-Mart: Learning Visual Action Trajectory Proposals for Manipulating 3D Articulated Objects——在清单中归入 **Action（Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action）**（清单未附方法摘要，此处不替原文概括贡献）。

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
    subgraph RL["RL 策略：交互式轨迹探索（仅训练期使用）"]
        INIT["随机采样关节体形状、任务 θ、起始部件位姿<br/>夹爪初始化在接触点 p 附近"]
        INIT --> TD3["条件 TD3 策略<br/>输入 33 维状态，输出残差夹爪位姿（下一个 waypoint）<br/>最多 5 个 waypoint"]
        TD3 --> ENV["仿真任务环境<br/>外在奖励：成功 / 引导 / 距离惩罚"]
        ENV --> TD3
    end
    ENV --> DATA["收集交互轨迹：成功样本为正例<br/>偏移任务 θ 构造负例"]
    subgraph PER["感知网络（测试时只用这部分）"]
        ENC["输入编码：PointNet++ 点云特征 f_s<br/>MLP 编码接触点 p、轨迹 τ、任务 θ"]
        ENC --> TS["Trajectory Scoring：MLP 预测轨迹成功概率 r"]
        ENC --> TP["Trajectory Proposal：cVAE 给出每点多样轨迹分布"]
        TP --> AP["Actionability Prediction：MLP 预测每点 actionability 分数<br/>目标为 100 条提议轨迹打分后的 top-5 均值"]
        TS --> AP
    end
    DATA --> ENC
    TS -.-> CUR["好奇心反馈：内在奖励 -500·r<br/>鼓励 RL 探索当前评分低的新轨迹"]
    CUR -.-> TD3
    AP --> OUT["输出：actionability 热力图 + 每点多样轨迹提议 + 成功评分"]
    TP --> OUT
    TS --> OUT
```

图依据：论文 §4 / Fig.2（RL 策略 §4.1，感知网络 §4.2，好奇心驱动探索 §4.3；输出定义见 §3）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ICLR 2022 |
| 方法名（清单标注） | VAT-Mart |
| 分组 1 | Action · Architecture: Hierarchical model；Affordance Usage: explicit；Action Subcategory: Hierarchical affordance-to-action |
| 论文 | <https://arxiv.org/abs/2106.14440> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；方法流程见上文「流程总览」，实验细节请读 [原文](https://arxiv.org/abs/2106.14440)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2106.14440) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「VAT-Mart」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_vat-mart-learning-visual-action-trajectory-proposals-for.md`](../../sources/papers/affordance_awesome_vat-mart-learning-visual-action-trajectory-proposals-for.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2106.14440>
- [arXiv PDF](https://arxiv.org/pdf/2106.14440)：已通读方法部分（实验与附录以略读为准）：§1 Introduction、§3 Actionable Visual Priors、§4 VAT-Mart（4.1–4.3）、§5 实验设置（略读）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2106.14440)
