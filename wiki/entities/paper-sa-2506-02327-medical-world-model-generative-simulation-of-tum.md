---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.02327"
venue: "arXiv 2025"
summary: "Generative simulation of tumor evolution for treatment planning."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_02327_medical-world-model-generative-simulatio.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Medical World Model

**Medical World Model: Generative Simulation of Tumor Evolution for Treatment Planning** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 521/571** 篇，分组 **84 Scientific Discovery**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Generative simulation of tumor evolution for treatment planning.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Generative simulation of tumor evolution for treatment planning.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    X["治疗前 CT x0 + 肿瘤 mask m0 + 语言治疗目标 g"] --> P["Policy Model：GPT-4o 读 CT 并按 TACE 指南生成动作库；DeepSeek-R1 精简药物集 D 与栓塞集 E，并给出临床规则"]
    P --> S["TACE Protocol Exploration：B 条并行 beam，先药物后栓塞，受 Hd / He 视野限制"]
    S --> F["Dynamics Model：Action-driven 3D 扩散模型 f_DM，生成 T 个治疗后 CT 样本"]
    F --> G["Assistant Model：微调 nnUNet 分割治疗后肝脏与肿瘤"]
    G --> H["Heuristic Function：3D ResNet + Cross-Attention 生存分析，输出风险分数"]
    H --> R["每步选风险最低的候选；淘汰最差 beam 并替换为最优 beam"]
    R --> S
    R --> O["输出风险最低的 TACE 方案及模拟治疗后 CT"]
    subgraph TRAIN["Dynamics Model 训练"]
        T1["放疗报告：GPT-4o + DeepSeek-R1 抽取关键词并生成动作描述"] --> T2["CLIP 文本编码 + 概念嵌入，融合为动作条件 τ(a)"]
        T3["Text-driven Morpho-Gaussian Attenuation：按报告疗效等级处理肿瘤 mask（仅训练时）"] --> T4["3D VQGAN latent 空间条件扩散"]
        T2 --> T4
        T5["Combo Contrastive Learning：同关键词动作拉近、异关键词推远"] --> T4
    end
    T4 -.-> F
```

图依据：论文 §3 Overall Framework（Fig. 2）/ §3.1 Policy / §3.2 Dynamics（Fig. 3）/ §3.3 Inverse Dynamics 与 Algorithm 1

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 521/571 |
| 分组 | 84 Scientific Discovery |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.02327> |

## 核心机制（归纳）

### 策展导读要点

Generative simulation of tumor evolution for treatment planning.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.02327) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **84 Scientific Discovery** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Medical World Model」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **84 Scientific Discovery**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World Models](../entities/awesome-world-models.md)
- 技术地图：[Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_wm_2506_02327_medical-world-model-generative-simulatio.md`](../../sources/papers/sun_awesome_wm_2506_02327_medical-world-model-generative-simulatio.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.02327>
- [arXiv PDF](https://arxiv.org/pdf/2506.02327)：已通读方法部分（实验与附录以略读为准）：§1 引言、§3.1–3.3 方法与 Algorithm 1；§4 实验仅略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.02327)
