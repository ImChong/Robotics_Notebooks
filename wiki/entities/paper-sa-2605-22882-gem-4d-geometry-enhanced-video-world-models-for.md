---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2605.22882"
venue: "arXiv 2025"
summary: "Geometry-enhanced video world models for robot manipulation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2605_22882_gem-4d-geometry-enhanced-video-world-mod.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# GEM-4D

**GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 299/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Geometry-enhanced video world models for robot manipulation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Geometry-enhanced video world models for robot manipulation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
    subgraph TR["训练: Geometry-Enhanced Velocity Alignment"]
        direction TB
        VID["视频 latent z (VAE 编码)"] --> DIT["Video DiT: 中层特征 m_t → 输出头 → 视频速度场"]
        VID --> GEO["冻结几何基础模型 G<br/>(如 VGGT / Depth Anything V3)提取稠密几何表示 g0"]
        DIT -->|"m_t 作为唯一场景级条件"| GDIT["Geometry DiT: 预测几何 latent 的速度场"]
        GEO -->|"加噪, 作为流匹配目标"| GDIT
        DIT --> L1["视频流匹配损失"]
        GDIT --> L2["几何流匹配损失 (α 加权)"]
    end
    TR -->|"推理仅用单流视频分支, 无额外开销"| ROLL["输入: 初始观测 I0 + 指令 c → 生成未来帧 rollout"]
    ROLL --> AIDS
    subgraph AIDS["Adaptive Inverse Dynamic System (AIDS)"]
        direction TB
        S1["3D 场景定位: Qwen3.5-VL + SAM 2 得目标/末端掩码<br/>FoundationPose 对齐末端 CAD 得初始位姿"] --> S2["置信度门控跟踪: CoTracker3 跟踪末端关键点<br/>漂移 → 重锚定; 骤降 → VLM 重新定位掩码"]
        S2 --> S3["位姿回退: FoundationPose 置信度低且时序跳变过大时<br/>平移取深度质心, 旋转做球面插值"]
        S3 --> S4["抓取插入与动作合成: GraspGen 候选按位姿偏差选优<br/>插值平滑 → 逆运动学"]
    end
    S4 --> ACT["6-DoF 机械臂可执行动作序列"]
```

图依据：论文 §3.1–3.3 / Fig.2、Fig.3、Fig.4

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 299/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2605.22882> |
| 项目页 | <https://anonymous-submission-20.github.io/gem.github.io/> |

## 核心机制（归纳）

### 策展导读要点

Geometry-enhanced video world models for robot manipulation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2605.22882) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「GEM-4D」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2605_22882_gem-4d-geometry-enhanced-video-world-mod.md`](../../sources/papers/sun_awesome_wm_2605_22882_gem-4d-geometry-enhanced-video-world-mod.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2605.22882>
- [arXiv PDF](https://arxiv.org/pdf/2605.22882)：已通读方法部分（实验与附录以略读为准）：§3 方法（3.1 问题形式化、3.2 Geometry-Enhanced Velocity Alignment、3.3 Adaptive Inverse Dynamic System）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2605.22882)
