---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2410.15461"
venue: "arXiv 2024"
code: https://github.com/litwellchi/EmbodiedVideoAnticipator?tab=readme-ov-file
summary: "Embodied world model for future video anticipation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2410_15461_eva-an-embodied-world-model-for-future-v.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# EVA

**EVA: An Embodied World Model for Future Video Anticipation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 290/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Embodied world model for future video anticipation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Embodied world model for future video anticipation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
    subgraph ARCH["EVA 结构 (§4.1)"]
        VLM["VLM 骨干 7B<br/>CLIP ViT-L/14 + Vicuna-v1.5<br/>token 聚类压缩视频 token"]
        AD["Cross-attention 生成适配器<br/>(VLM 隐状态 → VDM 文本嵌入)"]
        VDM["VDM 1.5B<br/>16 帧 / 2 秒视频扩散模型<br/>+ Ensemble-LoRA (按任务 token 门控)"]
        VLM --> AD --> VDM
    end
    subgraph TRAIN["三阶段训练 (§4.2)"]
        T1["阶段 1：COCO + CC3M-595K<br/>预热 VLM 视觉适配器"] --> T2["阶段 2：QA 指令微调<br/>开放域多模态指令 + EVA 指令数据"]
        T2 --> T3["阶段 3：整体适配<br/>训练生成适配器 + 各任务 Ensemble-LoRA"]
    end
    subgraph ROG["Reflection of Generation 推理 (§3, §4.3, Alg. 1)"]
        R0["输入：初始观测 O0 + 任务指令 I"] --> R1["VLM 编码隐状态 H<br/>经适配器条件化 VDM 生成视频块 v(0)"]
        R1 --> R2{"Reflect：自问<br/>任务是否完成?"}
        R2 -->|"否：Extend"| R3["以关键帧为条件<br/>继续生成扩展帧 v(k+1)"]
        R3 --> R2
        R2 -->|"Regenerate"| R1
        R2 -->|"是：Output"| R4["拼接 v(0)…v(k) 输出预测视频 + 文本回答"]
    end
    T3 -.->|"训练后的 EVA"| R1
    VDM -.-> R1
```

图依据：论文 §3 Reflection-of-Generation（式 2、Algorithm 1、Fig. 3）、§4.1 模型结构、§4.2 多阶段训练、§4.3 Autoregressive Frame Extension。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 290/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2410.15461> |
| 代码/项目 | <https://github.com/litwellchi/EmbodiedVideoAnticipator?tab=readme-ov-file> |

## 核心机制（归纳）

### 策展导读要点

Embodied world model for future video anticipation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2410.15461) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EVA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **812 Manipulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2410_15461_eva-an-embodied-world-model-for-future-v.md`](../../sources/papers/sun_awesome_wm_2410_15461_eva-an-embodied-world-model-for-future-v.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2410.15461>
- [arXiv PDF](https://arxiv.org/pdf/2410.15461)：已通读方法部分（实验与附录以略读为准）：§3 RoG、§4.1～4.3；附录 A 的模型/训练阶段略读。

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2410.15461)
