---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2408.16647"
venue: "arXiv 2024"
summary: "Real-world video generation for VLM-based autonomous driving."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2408_16647_drivegenvlm-real-world-video-generation.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# DriveGenVLM

**DriveGenVLM: Real-world Video Generation for VLM based Autonomous Driving** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 420/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Real-world video generation for VLM-based autonomous driving.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Real-world video generation for VLM-based autonomous driving.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["Waymo Open Dataset<br/>前/左前/右前三路相机视频（128×128，截取 175 帧）"] --> B["Flexible Diffusion Model (FDM)<br/>U-Net + DDPM，按帧子集条件训练"]
    B --> C["采样方案 sampling scheme：<br/>一系列 (X_s 待采样帧索引, Y_s 条件帧索引)"]
    C --> D["Autoreg：每步以前 10 帧为条件采样 10 帧"]
    C --> E["Hierarchy-2：先稀疏关键帧，再分组补全中间帧"]
    C --> F["Adaptive Hierarchy-2：测试时按 LPIPS 距离自适应选条件帧"]
    D --> G["生成后续驾驶视频"]
    E --> G
    F --> G
    G --> H["FVD 评估生成质量"]
    G --> I["预训练 EILEV (BLIP-2) 上下文学习<br/>为生成视频产生动作旁白"]
    I --> J["旁白用于检验视频可解释、可供 VLM 使用"]
```

图依据：论文 §II-A/B（DDPM、EILEV，Fig.2–3）、§III 方法（训练架构与采样方案，Table I）、§IV-E（EILEV 验证）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 420/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2408.16647> |

## 核心机制（归纳）

### 策展导读要点

Real-world video generation for VLM-based autonomous driving.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2408.16647) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「DriveGenVLM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2408_16647_drivegenvlm-real-world-video-generation.md`](../../sources/papers/sun_awesome_wm_2408_16647_drivegenvlm-real-world-video-generation.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2408.16647>
- [arXiv PDF](https://arxiv.org/pdf/2408.16647)：已通读方法部分（实验与附录以略读为准）：§I 引言、§II 背景（DDPM、EILEV 上下文学习）、§III 方法（训练架构、采样方案）、§IV 实验设定与 EILEV 验证

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2408.16647)
