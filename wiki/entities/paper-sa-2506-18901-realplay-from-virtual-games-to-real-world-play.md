---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.18901"
venue: "arXiv 2025"
code: https://github.com/wenqsun/Real-Play
summary: "From virtual games to real-world play."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_18901_realplay-from-virtual-games-to-real-worl.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# RealPlay

**RealPlay: From Virtual Games to Real-World Play** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 181/571** 篇，分组 **52 Game & Open-World Exploration Application**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

From virtual games to real-world play.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- From virtual games to real-world play.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    PRE["预训练图生视频模型 CogVideoX\nT5 文本编码 + 视频 VAE + DiT"] --> S1
    subgraph S1["阶段1：改造为分块（chunk-wise）生成模型"]
        direction TB
        A1["以前一视频块作为条件（替换首帧条件）"] --> A2["专用注意力掩码\n每块生成 4 个未来帧 latent（16 帧）"]
        A2 --> A3["训练时对条件块加噪（Noise Augmentation）"]
    end
    OV["通用视频数据 OpenViD 100K"] --> S1
    S1 --> S2
    GD["带标注游戏数据\n(块 C^k, 动作 a^k)，动作为前进/左转/右转 one-hot"] --> S2
    RD["无标注真实视频\n车辆 / 自行车 / 行人（动作输入置为全零向量）"] --> S2
    subgraph S2["阶段2：RealPlay 动作控制微调"]
        direction TB
        B1["动作 → MLP（512 维）+ 噪声时间步嵌入"] --> B2["线性层产生 adaLN 调制参数\n调制注意力与 FFN"]
    end
    S2 --> INF["推理：用户看到上一块 C^k，给出游戏动作指令\n生成下一块 C^k+1，迭代交互"]
```

图依据：论文 §3 / Fig.2、§3.1–3.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 181/571 |
| 分组 | 52 Game & Open-World Exploration Application |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.18901> |
| 代码/项目 | <https://github.com/wenqsun/Real-Play> |

## 核心机制（归纳）

### 策展导读要点

From virtual games to real-world play.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.18901) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **52 Game & Open-World Exploration Application** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「RealPlay」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **52 Game & Open-World Exploration Application**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2506_18901_realplay-from-virtual-games-to-real-worl.md`](../../sources/papers/sun_awesome_wm_2506_18901_realplay-from-virtual-games-to-real-worl.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.18901>
- [arXiv PDF](https://arxiv.org/pdf/2506.18901)：已通读方法部分（实验与附录以略读为准）：§3 方法（含 Fig.2、§3.1 分块生成、§3.2 从游戏到真实世界）、§4 开头的实现细节

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.18901)
