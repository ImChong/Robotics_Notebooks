---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2402.08268"
venue: "arXiv 2024"
code: https://github.com/LargeWorldModel/LWM
summary: "Million-length video and language world model based on RingAttention."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2402_08268_lwm-world-model-on-million-length-video.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# LWM

**LWM: World Model on Million-Length Video And Language With RingAttention** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 063/571** 篇，分组 **41 Language-Based World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Million-length video and language world model based on RingAttention.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Million-length video and language world model based on RingAttention.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  BASE["Llama-2 7B"] --> S1
  subgraph S1["阶段 I：长上下文语言模型 (§3)"]
    BK["Books3 书籍文本\n逐级加长: 32K → 128K → 256K → 512K → 1M"] --> RING["Blockwise RingAttention + FlashAttention\nRoPE θ 随上下文长度放大"]
    QA["模型生成的问答数据\n1000 token 分块生成问答后拼接成长序列"] --> CHAT["对话微调\n混合 UltraChat 与问答数据 (约 7:3)"]
    RING --> TXT["LWM-Text-1M"]
    TXT --> CHAT
  end
  TXT --> S2
  subgraph S2["阶段 II：长视频 + 语言联合训练 (§4)"]
    VQ["VQGAN 逐帧分词\n每帧 256 个 token，视觉用 vision 标记包裹"] --> MIX["文本与视觉 token 交错拼接\n自回归预测下一个 token"]
    ST["渐进训练: LWM-1K 文本-图像 → LWM-8K 文本-视频 → LWM-Chat-32K / 128K / 1M"] --> MIX
    PACK["masked sequence packing\n每个文本-视觉对只关注自身并重加权损失"] --> MIX
  end
  MIX --> OUT["LWM / LWM-Chat\n长视频理解、文本到图像 / 视频生成"]
```

图依据：论文 Method Overview、§3（阶段 I）、§4.1–4.2（阶段 II）、Fig. 3 / Fig. 4。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 063/571 |
| 分组 | 41 Language-Based World Models |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2402.08268> |
| 代码/项目 | <https://github.com/LargeWorldModel/LWM> |

## 核心机制（归纳）

### 策展导读要点

Million-length video and language world model based on RingAttention.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2402.08268) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **41 Language-Based World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「LWM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **41 Language-Based World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2402_08268_lwm-world-model-on-million-length-video.md`](../../sources/papers/sun_awesome_wm_2402_08268_lwm-world-model-on-million-length-video.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2402.08268>
- [arXiv PDF](https://arxiv.org/pdf/2402.08268)：已通读方法部分（实验与附录以略读为准）：Method Overview、§3 Stage I、§4.1–4.2 Stage II（评测与附录略读）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2402.08268)
