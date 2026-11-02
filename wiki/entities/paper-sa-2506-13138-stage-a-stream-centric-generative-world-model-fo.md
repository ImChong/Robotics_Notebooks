---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.13138"
venue: "arXiv 2025"
summary: "Stream-centric generative world model for long-horizon driving-scene simulation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_13138_stage-a-stream-centric-generative-world.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# STAGE

**STAGE: A Stream-Centric Generative World Model for Long-Horizon Driving-Scene Simulation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 481/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Stream-centric generative world model for long-horizon driving-scene simulation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Stream-centric generative world model for long-horizon driving-scene simulation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["输入：初始帧 I0 + HD map + 3D bounding box<br/>（锚帧 = I0 的 latent，条件帧 = 上一帧 I(T-1) 的 latent）"] --> GEN
    subgraph GEN["逐帧流式生成（Stable Diffusion U-Net，第 T 帧、第 t 步去噪）"]
        BUF["StreamingBuffer：FIFO 缓存前 N=10 帧特征<br/>（按去噪步各维护一份）"] --> SEL["Selector：取前第 1、5、10 帧特征"]
        SEL --> HTFT["HTFT 分层时间特征迁移：<br/>GroupNorm+Linear → CrossAttn → 融合特征 F_fused"]
        HTFT --> UN["引导 U-Net 去噪 → 第 T 帧 latent → 解码为图像 I(T)"]
        UN -->|更新缓存| BUF
    end
    GEN --> OUT["长时程驾驶视频 I0 … I(T)"]
    subgraph INF["无限生成（数据集条件用尽时）"]
        MILE["MILE 预测自车朝向/速度/加速度<br/>与前视 BEV 标签 L(t+1)"] --> PRJ["提取车道线/人行横道 → HD map；人/车 → bbox<br/>经 T(ego→cam) 与内参 K 投影到前视"]
    end
    UN --> MILE
    PRJ -->|作为下一帧条件| IN
    subgraph TR["三阶段训练"]
        S1["Stage 1 流式学习：关闭 HTFT"] --> S2["Stage 2 HTFT 学习：冻结 HTFT 之前参数，缓存特征"] --> S3["Stage 3 模拟推理：条件帧换成推理生成的图像"]
    end
```

图依据：论文 §III-A–D / Fig.2（含 StreamingBuffer、HTFT 公式、三阶段训练、MILE 无限生成）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 481/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.13138> |

## 核心机制（归纳）

### 策展导读要点

Stream-centric generative world model for long-horizon driving-scene simulation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.13138) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「STAGE」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2506_13138_stage-a-stream-centric-generative-world.md`](../../sources/papers/sun_awesome_wm_2506_13138_stage-a-stream-centric-generative-world.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.13138>
- [arXiv PDF](https://arxiv.org/pdf/2506.13138)：已通读方法部分（实验与附录以略读为准）：§III-A–D 方法（含训练策略与无限生成）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.13138)
