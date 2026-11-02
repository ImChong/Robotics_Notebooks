---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2603.25175"
venue: "arXiv 2026"
summary: "Action-conditioned motion modeling + kinematic joint embedding for robust wearable camera pose estimation"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2603_25175_ag-egopose-leveraging-action-guided-moti.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# AG-EgoPose

**AG-EgoPose: Leveraging Action-Guided Motion and Kinematic Joint Encoding for Egocentric 3D Pose Estimation** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 205/249** 篇，分组 **Human Pose & Mesh Recovery**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Action-conditioned motion modeling + kinematic joint embedding for robust wearable camera pose estimation

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Action-conditioned motion modeling + kinematic joint embedding for robust wearable camera pose estimation
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    CLIP["第一人称 RGB 视频片段"] --> HM["冻结 2D 热图网络<br/>ConvNeXt-Tiny + FPN，输出 J=15 关节热图与 96 通道 FPN 特征"]
    HM --> TOKEN["热图 token 化：软质心处采样 FPN 特征<br/>融合为关节 token s0"]
    TOKEN --> SJT["Spatial Joint Transformer + MLP 头"]
    SJT --> PSP["空间姿态 P_sp（独立监督，作为锚点）"]
    CLIP --> DINO["冻结 DINOv2 ViT-S/14"]
    HM -->|"热图作软空间 query"| POOL["热图引导 DINO 池化<br/>得关节局部外观特征 l"]
    DINO --> POOL
    DINO -->|"CLS token 序列"| AF["Ego4D 预训练 ActionFormer<br/>多尺度融合为时间动作上下文 A<br/>仅微调投影层与最后两个块"]
    SJT --> Q["关节 query q = LN(s + f(l))"]
    POOL --> Q
    Q --> RET["逐关节时间检索：query 对 A 做交叉注意力<br/>MLP 预测残差 ΔP（末层零初始化）"]
    AF --> RET
    HM -->|"热图统计量：质心、方差、熵、峰值等"| GATE["小 MLP 得有界门 α ∈ [0.05, 0.80]"]
    PSP --> OUT["P = P_sp + α·ΔP"]
    RET --> OUT
    GATE --> OUT
```

图依据：论文 §3.1–3.3 / Fig.1、Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 205/249 |
| 分组 | Human Pose & Mesh Recovery |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2603.25175> |

## 核心机制（归纳）

### 策展导读要点

Action-conditioned motion modeling + kinematic joint embedding for robust wearable camera pose estimation

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2603.25175) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Human Pose & Mesh Recovery** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「AG-EgoPose」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Human Pose & Mesh Recovery**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2603_25175_ag-egopose-leveraging-action-guided-moti.md`](../../sources/papers/sun_awesome_ego_2603_25175_ag-egopose-leveraging-action-guided-moti.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2603.25175>
- [arXiv PDF](https://arxiv.org/pdf/2603.25175)：已通读方法部分（实验与附录以略读为准）：§3（Spatial Pose Anchor、Joint-Local Appearance and Action Context、Heatmap-Statistics-Gated Residual Decoder、Training Objective）、§4.2 实现细节开头；实验结果仅略读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2603.25175)
