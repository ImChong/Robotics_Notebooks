---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2606.07326"
venue: "arXiv 2026"
summary: "Anchor-keyframe constrained controllable egocentric simulator, with anchor matching accuracy reaching 89%"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2606_07326_anchorworld-embodied-egocentric-world-si.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# AnchorWorld

**AnchorWorld: Embodied Egocentric World Simulation with View-based Evolution Customization** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 103/249** 篇，分组 **28 Egocentric World Models**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Anchor-keyframe constrained controllable egocentric simulator, with anchor matching accuracy reaching 89%

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Anchor-keyframe constrained controllable egocentric simulator, with anchor matching accuracy reaching 89%
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["初始第一人称帧 I0"] --> V["Wan2.2 DiT 视频生成骨干（flow matching）"]
    M["SMPL-X 人体动作 M"] --> ME["motion encoder → 动作 token z_m"]
    C["相机位姿（训练时先外部视角，后头部第一人称视角）"] --> CE["camera encoder → 位姿 token z_c"]
    ME --> SPA["Spatial Pose Attention：与视频 token 拼接做自注意力，之后截断辅助 token"]
    CE --> SPA
    SPA --> V
    S["锚点视图 (RGB 图像 Ii, 6-DoF 位姿 ci, 演化提示 ti)"] --> IMG["图像 latent 沿帧维拼接（in-context）+ 3D RoPE"]
    S --> POSE["View Pose Injection：位姿嵌入加到视觉 token"]
    S --> TXT["演化提示 ti：带注意力掩码的 cross-attention"]
    IMG --> V
    POSE --> V
    TXT --> V
    V --> OUT["第一人称世界模拟视频"]
    subgraph TRAIN["渐进式四阶段训练"]
        T1["Stage I：第三人称动作训练"] --> T2["Stage II：第一人称动作训练"] --> T3["Stage III：静态锚点视图定制"] --> T4["Stage IV：动态锚点视图演化"]
    end
```

图依据：论文 §3 方法（3.1 混合视角动作控制、3.2 可演化锚点视图定制、3.3 渐进式多阶段训练）/ Fig.2、Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 103/249 |
| 分组 | 28 Egocentric World Models |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.07326> |

## 核心机制（归纳）

### 策展导读要点

Anchor-keyframe constrained controllable egocentric simulator, with anchor matching accuracy reaching 89%

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.07326) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **28 Egocentric World Models** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「AnchorWorld」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **28 Egocentric World Models**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md)
- 技术地图：[Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_ego_2606_07326_anchorworld-embodied-egocentric-world-si.md`](../../sources/papers/sun_awesome_ego_2606_07326_anchorworld-embodied-egocentric-world-si.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2606.07326>
- [arXiv PDF](https://arxiv.org/pdf/2606.07326)：已通读方法部分（实验与附录以略读为准）：摘要、§3 方法（含 3.1–3.3）、§4.1 实验设定开头；其余实验与附录略读或未读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2606.07326)
