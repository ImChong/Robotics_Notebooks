---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2607.08537"
venue: "ECCV 2026"
summary: "The first learning-based solution to the 'Out of Sight, Not out of Mind' (OSNOM) task. A transformer-based model with an updatable memory of established tracks and a track assignment module. Trained on 56 videos, achieve"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2607_08537_whareformer-learning-to-track-what-is-wh.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Whareformer

**Whareformer: Learning to Track What is Where in Long Egocentric Videos** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 220/249** 篇，分组 **Human Pose & Mesh Recovery**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

The first learning-based solution to the 'Out of Sight, Not out of Mind' (OSNOM) task. A transformer-based model with an updatable memory of established tracks and a track assignment module. Trained on 56 videos, achieves SOTA on 260 long test videos from EPIC-KITCHENS-100, IT3DEgo, and HD-EPIC

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- The first learning-based solution to the 'Out of Sight, Not out of Mind' (OSNOM) task. A transformer-based model with an updatable memory of established tracks and a track assignment module. Trained on 56 videos, achieves SOTA on 260 long test videos from EPIC-KITCHENS-100, IT3DEgo, and HD-EPIC
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    VID["第一人称长视频<br/>逐帧得到物体分割 mask"] --> OBS["构造观测 o = (a, l)"]
    OBS --> A["外观 a：DINOv2 特征（用 mask 隔离物体）<br/>ℓ2 归一化 → PCA 降维 → 再 ℓ2 归一化"]
    OBS --> L["3D 位置 l：单目深度对齐 3D 重建<br/>把物体中心抬升到世界坐标"]
    MEM["Track 记忆<br/>外观：DenStream 在线聚类（persistent + transient 簇）<br/>位置：最近 W 个 3D 位置缓冲（约 1 秒）"] --> DIST
    A --> DIST["对每个现有 track 计算距离<br/>外观距离 c^A、位置距离 c^L"]
    L --> DIST
    DIST --> EMB["拼接后线性投影 g → 嵌入 e_n,k"]
    EMB --> SEQ["输入序列 S = [NT, e_1, …, e_T]<br/>New Track token 置于最前，无位置编码"]
    SEQ --> TF["Transformer encoder<br/>所有 token 互相注意"]
    TF --> HEAD["线性分类头 + softmax<br/>分布覆盖已有 track 与新 track"]
    HEAD --> CF["argmax 分配；多个观测选中同一 track 时<br/>按置信度迭代解决冲突"]
    CF --> UPD["更新 track 记忆<br/>已有 track：追加位置、更新 DenStream 簇<br/>新 track：用该观测初始化"]
    UPD --> MEM
```

图依据：论文 §3.2–3.7 / Fig.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 220/249 |
| 分组 | Human Pose & Mesh Recovery |
| 出处 | ECCV 2026 |
| 论文 | <https://arxiv.org/abs/2607.08537> |

## 核心机制（归纳）

### 策展导读要点

The first learning-based solution to the 'Out of Sight, Not out of Mind' (OSNOM) task. A transformer-based model with an updatable memory of established tracks and a track assignment module. Trained on 56 videos, achieves SOTA on 260 long test videos from EPIC-KITCHENS-100, IT3DEgo, and HD-EPIC

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.08537) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Human Pose & Mesh Recovery** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Whareformer」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_ego_2607_08537_whareformer-learning-to-track-what-is-wh.md`](../../sources/papers/sun_awesome_ego_2607_08537_whareformer-learning-to-track-what-is-wh.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2607.08537>
- [arXiv PDF](https://arxiv.org/pdf/2607.08537)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§3 Whareformer（3.1–3.7）、§4.1–4.3 实验设定与结果

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2607.08537)
