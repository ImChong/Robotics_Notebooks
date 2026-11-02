---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-10-23
project_id: aff-affordance-learning-from-play-for-sample-efficient-polic
arxiv: "2203.00352"
venue: "ICRA 2022"
summary: "ICRA 2022 的 affordance 论文，清单分组：Action（Architecture: E2E；Affordance Usage: reward；Action Subcategory: Integrated policy · Optimization signal）。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/affordance_awesome_affordance-learning-from-play-for-sample-efficient-polic.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# VAPO

**Affordance Learning from Play for Sample-Efficient Policy Learning**（ICRA 2022）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套论文清单。本页是 **清单索引**：给出它在综述分类里的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Affordance Learning from Play for Sample-Efficient Policy Learning——在清单中归入 **Action（Architecture: E2E；Affordance Usage: reward；Action Subcategory: Integrated policy · Optimization signal）**（清单未附方法摘要；方法流程见「流程总览」，贡献概括以原文为准）。

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
    P["人类遥操作 play 数据<br/>静态相机 + 夹爪相机，无人工标注"] --> L["自监督标注：夹爪闭合/张开时刻<br/>投影 3D TCP 到图像，半径内像素标为 affordance 区域<br/>并生成指向区域中心的方向向量"]
    L --> FA["训练 affordance 模型 F_a（U-Net + 两路卷积分支）<br/>输出二值 mask A 与中心方向向量<br/>Hough voting 得到中心"]
    FA --> S["静态相机 affordance 模型"]
    FA --> G["夹爪相机 affordance 模型"]
    S --> C["预测 affordance 区域中心，定位到 3D"]
    C --> M["π_mod 基于模型的规划<br/>把末端从自由空间移到 affordance 区域附近"]
    M --> SW{"按末端到中心的归一化距离 α(s) 切换"}
    SW --> RL["π_rl 局部 RL 策略（SAC，夹爪相机）<br/>输入：RGB-D、affordance mask、本体状态、到中心距离"]
    G --> RL
    RL --> R["奖励 r = λ1 R_succ + λ2 R_aff + λ3 R_out<br/>R_aff 鼓励靠近 affordance 中心"]
    R --> RL
    RL --> ACT["动作：delta XYZ、delta 欧拉角、夹爪开合"]
```

图依据：论文 §III / Fig.1、Fig.3（策略混合式 (1) 见 §III，affordance 学习见 §III-A，规划到 RL 切换见 §III-B，奖励式 (2) 见 §III-C，网络细节见 §IV）

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | ICRA 2022 |
| 方法名（清单标注） | VAPO |
| 分组 1 | Action · Architecture: E2E；Affordance Usage: reward；Action Subcategory: Integrated policy · Optimization signal |
| 论文 | <https://arxiv.org/abs/2203.00352> |

## 核心机制（归纳）

清单本身只给出标题、出处与分类标签；方法流程见上文「流程总览」，实验细节请读 [原文](https://arxiv.org/abs/2203.00352)。

## 评测与指标

- 本页没有搬运原文的 benchmark 与实机指标；评测口径以 [原文](https://arxiv.org/abs/2203.00352) 为准。
- 综述本身不提供跨论文性能排名（见 [综述实体页](../entities/paper-affordance-learning-embodied-ai-survey.md)），不同设置下的数字不可直接比较。

## 与其他工作对比

- 本页不做与具体基线的数值对比；同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。
- 若站内另有这篇的深读页，以那一页为准。

## 结论

**本页只提供「VAPO」在 affordance 综述清单中的坐标与原文入口，不能替代原文。**

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

- [`sources/papers/affordance_awesome_affordance-learning-from-play-for-sample-efficient-polic.md`](../../sources/papers/affordance_awesome_affordance-learning-from-play-for-sample-efficient-polic.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://arxiv.org/abs/2203.00352>
- [arXiv PDF](https://arxiv.org/pdf/2203.00352)：已通读方法部分（实验与附录以略读为准）：§I Introduction、§III Approach（A–C）、§IV Implementation Details、§V 实验设置（略读）

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://arxiv.org/abs/2203.00352)
