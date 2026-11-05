---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2506.09981"
venue: "arXiv 2025"
code: https://github.com/OpenDriveLab/ReSim
summary: "Reliable world simulation for autonomous driving."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2506_09981_resim-reliable-world-simulation-for-auto.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# ReSim

**ReSim: Reliable World Simulation for Autonomous Driving** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 475/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Reliable world simulation for autonomous driving.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Reliable world simulation for autonomous driving.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph DATA["异构数据（§2.1）"]
        OD["OpenDV 网络驾驶视频（约 4M）"]
        NV["NAVSIM 专家动作（约 85K）"]
        CA["CARLA 仿真：PDM-Lite 专家 + 随机转向/速度的非专家探索（约 88K）"]
    end
    DATA --> TRAIN
    subgraph WM["可控世界模型 ReSim（§2.2，基于 CogVideoX）"]
        TRAIN["三阶段渐进训练：<br/>1 OpenDV（历史帧+文本指令）<br/>2 +NAVSIM/CARLA，冻结主干，仅训轨迹编码器+LoRA<br/>3 高分辨率全量微调"] --> RS["ReSim：9 帧历史 + 可选指令 c + 未来自车路点 a<br/>→ 扩散 Transformer → 未来视频"]
        LOSS["损失：视频扩散损失 + 0.1×动力学一致性损失<br/>偏向高噪声步采样"] -.-> TRAIN
    end
    subgraph V2R["Video2Reward（§2.3）"]
        DI["冻结 DINOv2 + 轻量预测头"] --> RW["视频 → 奖励<br/>（以 CARLA infraction score 监督）"]
    end
    CA -.->|监督信号| DI
    subgraph APP["应用（§2.4）"]
        P1["视频预测策略：无动作条件想象未来帧 → 逆动力学模型 IDM → 自车轨迹"]
        P2["奖励引导策略选择：候选轨迹 → ReSim 生成视频 → V2R 打分 → 选最高"]
        P3["闭环视觉仿真：智能体动作 → ReSim 生成下一视觉状态 → 反馈给智能体"]
    end
    RS --> P1
    RS --> P2
    RW --> P2
    RS --> P3
```

图依据：论文 §2.1–2.4 / Fig.2（Video2Reward）、Fig.3（视频预测策略）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 475/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2506.09981> |
| 代码/项目 | <https://github.com/OpenDriveLab/ReSim> |

## 核心机制（归纳）

### 策展导读要点

Reliable world simulation for autonomous driving.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2506.09981) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ReSim」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2506_09981_resim-reliable-world-simulation-for-auto.md`](../../sources/papers/sun_awesome_wm_2506_09981_resim-reliable-world-simulation-for-auto.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2506.09981>
- [arXiv PDF](https://arxiv.org/pdf/2506.09981)：已通读方法部分（实验与附录以略读为准）：§2.1 数据、§2.2 可控世界模型、§2.3 Video2Reward、§2.4 应用

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2506.09981)
