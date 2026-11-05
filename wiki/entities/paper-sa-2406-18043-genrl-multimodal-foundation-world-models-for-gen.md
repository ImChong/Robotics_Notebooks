---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2406.18043"
venue: "arXiv 2024"
code: https://github.com/mazpie/genrl
summary: "Multimodal foundation world models for generalist embodied agents."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2406_18043_genrl-multimodal-foundation-world-models.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# GenRL

**GenRL: Multimodal foundation world models for generalist embodied agents** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 258/571** 篇，分组 **73 Offline & Generalization**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Multimodal foundation world models for generalist embodied agents.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Multimodal foundation world models for generalist embodied agents.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph WM["世界模型 §3.1"]
        A["视觉观测 xt"] --> B["Encoder 编码为离散潜状态 st"]
        B --> C["Sequence model GRU 得到 ht"]
        C --> D["Dynamics predictor 预测 st"]
        B --> E["Decoder 重建 xt"]
    end
    subgraph AL["连接 VLM 与世界模型 §3.2"]
        F["InternVideo2 视频嵌入 e(v)<br/>k=8 帧"] --> G["Connector<br/>由嵌入预测潜状态 KL 对齐 Encoder 输出"]
        H["Aligner<br/>把 e(v) 邻域加噪点映射回 e(v)<br/>只用视觉数据 无语言标注"] --> G
    end
    B -.训练目标.-> G
    P["任务提示 语言或视频"] --> I["VLM 嵌入器得到 etask"]
    I --> J["Aligner + Connector<br/>生成目标潜状态轨迹"]
    G --> J
    J --> K["最佳匹配轨迹 时间对齐<br/>滑动比较前 b=8 个状态"]
    D --> L["策略 在想象中 rollout 潜状态"]
    L --> K
    K --> M["奖励 Eq.3<br/>线性投影后潜状态的余弦相似度"]
    M --> N["Actor-critic 在想象中训练策略"]
    N --> L
```

图依据：论文 §3.1–§3.3 / Fig.2、Fig.3、Eq.1–3（视频提示是否绕过 Aligner 论文称“可按需不用”，图中按语言路径画出）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 258/571 |
| 分组 | 73 Offline & Generalization |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2406.18043> |
| 代码/项目 | <https://github.com/mazpie/genrl> |

## 核心机制（归纳）

### 策展导读要点

Multimodal foundation world models for generalist embodied agents.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2406.18043) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **73 Offline & Generalization** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「GenRL」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **73 Offline & Generalization**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2406_18043_genrl-multimodal-foundation-world-models.md`](../../sources/papers/sun_awesome_wm_2406_18043_genrl-multimodal-foundation-world-models.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2406.18043>
- [arXiv PDF](https://arxiv.org/pdf/2406.18043)：已通读方法部分（实验与附录以略读为准）：§3.1 世界模型、§3.2 多模态基础世界模型（Connector / Aligner）、§3.3 想象中学习任务与时间对齐；§4 实验与附录以略读/未读为准


## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2406.18043)
