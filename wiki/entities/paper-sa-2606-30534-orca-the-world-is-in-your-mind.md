---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2606.30534"
venue: "arXiv 2026"
summary: "BAAI's general world foundation model learning a unified latent space from multimodal signals; pre-trained on 125K hours of video and 160M event annotations."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2606_30534_orca-the-world-is-in-your-mind.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Orca

**Orca: The World is in Your Mind** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 084/571** 篇，分组 **42 Visual / Video World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

BAAI's general world foundation model learning a unified latent space from multimodal signals; pre-trained on 125K hours of video and 160M event annotations.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- BAAI's general world foundation model learning a unified latent space from multimodal signals; pre-trained on 125K hours of video and 160M event annotations.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph PRE["预训练：Orca Encoder（原生预训练 VLM + 可学习 Query）"]
        direction TB
        VID["视频帧 vt（Video Data）"] --> O1["1) Observation-only 状态转移（unconscious）：Query 1 末层隐状态经两层 MLP 预测下一帧潜变量"]
        EVT["相邻事件描述 e(t+Δ)（Event Data）"] --> O2["2) Event-conditioned 状态转移（conscious）：Query 2 末层隐状态经两层 MLP 预测目标帧潜变量"]
        VID --> O2
        VQA["视频 V + 问题 lq（VQA Data）"] --> O3["3) VQA 回答生成（conscious）：LM head 自回归输出答案 la"]
        FV["冻结视觉编码器：给出目标帧潜变量作 teacher forcing"] -.-> O1
        FV -.-> O2
    end
    O1 --> LAT["统一世界潜空间 World Latent"]
    O2 --> LAT
    O3 --> LAT
    LAT --> FRZ["预训练后 backbone 冻结，仅训练轻量读出模块"]
    FRZ --> R1["到语言：复用 LM head 生成文本"]
    FRZ --> R2["到视觉：MLP adaptor + LoRA（冻结 SD3.5）多步去噪生成图像"]
    FRZ --> R3["到动作：MLP adaptor + DiT Action Expert（flow matching），输入含 q1 潜变量、本体感觉、带噪动作，输出 Action Chunk"]
```

图依据：论文 §2.1–2.2 / Fig. 1–2、§3.1.1、§3.2 / Fig. 4

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 084/571 |
| 分组 | 42 Visual / Video World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.30534> |

## 核心机制（归纳）

### 策展导读要点

BAAI's general world foundation model learning a unified latent space from multimodal signals; pre-trained on 125K hours of video and 160M event annotations.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.30534) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **42 Visual / Video World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Orca」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **42 Visual / Video World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2606_30534_orca-the-world-is-in-your-mind.md`](../../sources/papers/sun_awesome_wm_2606_30534_orca-the-world-is-in-your-mind.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2606.30534>
- [arXiv PDF](https://arxiv.org/pdf/2606.30534)：已通读方法部分（实验与附录以略读为准）：§2 Orca（建模与架构）、§3.1 预训练配方与数据、§3.2 下游后训练读出

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2606.30534)
