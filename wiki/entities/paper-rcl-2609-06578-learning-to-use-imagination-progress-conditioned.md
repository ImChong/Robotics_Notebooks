---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2609.06578"
venue: "2026"
summary: "ProWAM learns execution progress from recent action–observation feedback and recurrent memory, then uses that state to redistribute imagined-future attention inside a joint video–action policy. Its strongest evidence com"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2609_06578_learning-to-use-imagination-progress-con.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Learning to Use Imagination

**Learning to Use Imagination: Progress-Conditioned Future Utilization for World Action Models** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 416/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

ProWAM learns execution progress from recent action–observation feedback and recurrent memory, then uses that state to redistribute imagined-future attention inside a joint video–action policy. Its strongest evidence combines executed manipulation results with ablations separating progress estimation from future uti...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- ProWAM learns execution progress from recent action–observation feedback and recurrent memory, then uses that state to redistribute imagined-future attention inside a joint video–action policy. Its strongest evidence combines executed manipulation results with ablations separating progress estimation from future uti...
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph S1["阶段 1：训练 SS-DTPE（自监督，随后冻结）"]
        L1["示范轨迹的时间顺序 + 指令前缀作为天然监督"] --> L2["损失：语义进度对齐 L_sem + 单调顺序 L_ord + 时间对比 L_temp"]
    end
    subgraph SSD["SS-DTPE：双时间尺度进度编码器"]
        A["当前观测 O_t + 上一动作 a_t-1 + 指令 L"] --> B["短期 Action-Observation 编码器<br/>FiLM + 双向 Cross-Attention → 证据 E_t"]
        C["上一步进度 tokens Z_t-1"] --> D["长期 Mamba 进度聚合器<br/>得记忆过滤表示"]
        B --> E["Progress Attention：进度状态吸收短期证据"]
        D --> E
        E --> F["进度 tokens Z_t"]
    end
    subgraph HP["HPIM：层次化进度条件想象调制（每个 MoT 层）"]
        G["Inter-progress 全局门 d_t"]
        H["Intra-progress 未来 latent 相关性 ω_t,f"]
        I["中间层加强的层系数 η_l"]
        G --> J["调制权重 β = η_l·((1 − d_t) + d_t·ω)"]
        H --> J
        I --> J
    end
    F --> G
    F --> H
    J --> K["作用于 MoT 中 action token → 未来 latent 的注意力 logit<br/>加 log β"]
    subgraph WAM["联合视频–动作 WAM 骨干（MoT）"]
        V["未来 latent 流（video expert）"] --> K
        W["action 流（action expert）"] --> K
        K --> X["ActHead 输出动作块 Â_t"]
    end
    X --> Y["以滚动时域方式执行；新执行的动作用于下一次进度更新"]
    Y -.-> A
    L2 -.-> SSD
    SSD -.->|"阶段 2：冻结 SS-DTPE，训练 WAM + HPIM"| HP
```

图依据：论文 §4.1–4.4 / Fig.3、Fig.4（式 6–36）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 416/564 |
| 分组 | WAMs |
| 出处 | 2026 |
| 论文 | <https://arxiv.org/abs/2609.06578> |
| 子类 / 象限 | 联合视频动作建模 · 记忆与长时序 |

## 核心机制（归纳）

### 策展导读要点

ProWAM learns execution progress from recent action–observation feedback and recurrent memory, then uses that state to redistribute imagined-future attention inside a joint video–action policy. Its strongest evidence combines executed manipulation results with ablations separating progress estimation from future uti...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2609.06578) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Learning to Use Imagination」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **WAMs**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2609_06578_learning-to-use-imagination-progress-con.md`](../../sources/papers/rcl_awesome_wam_2609_06578_learning-to-use-imagination-progress-con.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2609.06578>
- [arXiv PDF](https://arxiv.org/pdf/2609.06578)：已通读方法部分（实验与附录以略读为准）：摘要、§1、§3.2 诊断分析、§4.1–4.4 方法与训练推理流程、§5.1 实现细节；其余实验与附录略读

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2609.06578)
