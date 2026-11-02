---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2607.09169"
venue: "arXiv 2026"
summary: "A temporally guided stereo framework coupling short-term motion evidence with projection-guided feature sampling. Achieves SOTA on UnrealEgo2 and UnrealEgo-RW"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2607_09169_tsr-ego-temporally-guided-stereo-refinem.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# TSR-Ego

**TSR-Ego: Temporally Guided Stereo Refinement Framework for Egocentric 3D Human Pose Estimation** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 219/249** 篇，分组 **Human Pose & Mesh Recovery**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A temporally guided stereo framework coupling short-term motion evidence with projection-guided feature sampling. Achieves SOTA on UnrealEgo2 and UnrealEgo-RW

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- A temporally guided stereo framework coupling short-term motion evidence with projection-guided feature sampling. Achieves SOTA on UnrealEgo2 and UnrealEgo-RW
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    I["因果窗口内的双目鱼眼图像<br/>左/右视图，仅用当前及过去帧"] --> E["Stereo Visual Encoder<br/>ResNet-18 + FPN → stride-4 稠密特征 F"]
    E --> T["Causal Temporal Feature Mixer (TFM)<br/>沿时间轴的因果深度可分离 1D 卷积<br/>不混合空间位置 → 特征 F̃"]
    Q["可学习关节 query Q0<br/>+ 可学习 3D 参考位姿 P0"] --> D1
    subgraph DEC["Causal Stereo Decoder（L 层重复）"]
        D1["因果时间自注意力<br/>对每个关节跨时间，含 age embedding"] --> D2["关节自注意力<br/>每帧内 J 个关节"]
        D2 --> D3["鱼眼投影当前 3D 位姿 → 各视图 2D 参考点<br/>可变形立体交叉注意力采样 F̃，出视野视图置零"]
        D3 --> D4["FFN 精炼"]
        D4 --> H["本层 Pose Head：P̂ℓ = P0 + hℓ(LN(Qℓ))"]
    end
    T --> D3
    H -- "detach 后作为下一层投影参考" --> D3
    H --> O["最终层输出绝对 3D 位姿<br/>各层均以 MPJPE 监督"]
```

图依据：论文 §3 / Fig.1（§3.1–3.6）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 219/249 |
| 分组 | Human Pose & Mesh Recovery |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2607.09169> |

## 核心机制（归纳）

### 策展导读要点

A temporally guided stereo framework coupling short-term motion evidence with projection-guided feature sampling. Achieves SOTA on UnrealEgo2 and UnrealEgo-RW

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.09169) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Human Pose & Mesh Recovery** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「TSR-Ego」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_ego_2607_09169_tsr-ego-temporally-guided-stereo-refinem.md`](../../sources/papers/sun_awesome_ego_2607_09169_tsr-ego-temporally-guided-stereo-refinem.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2607.09169>
- [arXiv PDF](https://arxiv.org/pdf/2607.09169)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§3 Method（3.1–3.6）、Fig.1

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2607.09169)
