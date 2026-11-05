---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2507.13162"
venue: "arXiv 2025"
code: https://lmb-freiburg.github.io/orbis.github.io/
summary: "Overcomes challenges of long-horizon prediction in driving world models."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2507_13162_orbis-overcoming-challenges-of-long-hori.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Orbis

**Orbis: Overcoming Challenges of Long-Horizon Prediction in Driving World Models** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 470/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Overcomes challenges of long-horizon prediction in driving world models.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Overcomes challenges of long-horizon prediction in driving world models.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
    IMG["输入图像"] --> TOK
    subgraph TOK["Hybrid Image Tokenizer（§4.1）"]
        direction TB
        ES["语义编码器 E_s<br/>由冻结 DINOv2 蒸馏（余弦相似度损失，仅训练时）"]
        ED["细节编码器 E_d"]
        QS["VQ 码本 C_s"]
        QD["VQ 码本 C_d"]
        ES --> QS
        ED --> QD
        CAT["拼接语义 + 细节表示<br/>连续 x=(x_s; x_d) 或量化 q=(q_s; q_d)<br/>微调时以 50% 概率绕过 VQ"]
        ES --> CAT
        ED --> CAT
        QS --> CAT
        QD --> CAT
    end
    TOK --> CTX["上下文帧潜变量 x_0:N-1"]
    EGO["（可选）ego-motion：转向角 + 速度<br/>两层 MLP 嵌入，经 AdaLN 注入"] --> WM
    CTX --> WM["世界模型：STDiT 块<br/>（空间注意力 + 时间注意力 + 前馈）<br/>目标帧初始为噪声或全 mask"]
    WM --> LOOP["M 步迭代去噪（flow matching，连续）<br/>或迭代 unmask（MaskGIT 式，离散）"]
    LOOP --> NEXT["预测下一帧潜变量"]
    NEXT --> DEC["Tokenizer 解码器 → 图像"]
    NEXT -- "滑动窗口：追加到上下文、丢弃最早一帧<br/>自回归 rollout" --> CTX
```

图依据：论文 §4.1–§4.2 / Fig.3、式 (1)–(4)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 470/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2507.13162> |
| 代码/项目 | <https://lmb-freiburg.github.io/orbis.github.io/> |

## 核心机制（归纳）

### 策展导读要点

Overcomes challenges of long-horizon prediction in driving world models.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2507.13162) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Orbis」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **82 Autonomous Driving**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2507_13162_orbis-overcoming-challenges-of-long-hori.md`](../../sources/papers/sun_awesome_wm_2507_13162_orbis-overcoming-challenges-of-long-hori.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2507.13162>
- [arXiv PDF](https://arxiv.org/pdf/2507.13162)：已通读方法部分（实验与附录以略读为准）：§4 方法（§4.1 Hybrid Image Tokenizer、§4.2 潜空间世界模型含 flow matching / masked generative / ego-motion 条件，Fig.3），§5 实验设定略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2507.13162)
