---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2603.27449"
venue: "arXiv 2026"
summary: "An egocentric world model that generates realistic human-object interactions as videos conditioned on an input image, a text prompt, and per-frame human actions, including both body poses and hand gestures"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2603_27449_lome-learning-human-object-manipulation.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# LOME

**LOME: Learning Human-Object Manipulation with Action-Conditioned Egocentric World Model** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 093/249** 篇，分组 **27 Embodied AI & Vision-Language-Action Ego-VLA**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

An egocentric world model that generates realistic human-object interactions as videos conditioned on an input image, a text prompt, and per-frame human actions, including both body poses and hand gestures

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- An egocentric world model that generates realistic human-object interactions as videos conditioned on an input image, a text prompt, and per-frame human actions, including both body poses and hand gestures
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IMG["输入图像 I"] --> ENC["预训练 VAE 编码器 E<br/>得到无噪声图像潜变量 x^I"]
    TXT["文本描述 y（umT5 编码）"] --> DIT
    CAM["相机外参 ζ + 内参 K"] --> RAY["Plücker 射线图 + 轻量相机适配器 C<br/>得到相机特征 z"]
    ACT["逐帧 3D 人体/手部关键点 P_i"] --> PROJ["投影到图像平面并栅格化<br/>去除视锥外关键点，得 2D 动作图 Â"]
    PROJ --> AENC["VAE 编码得动作潜变量 a"]
    RAY --> CAT
    ENC --> CAT["沿时间维拼接<br/>[x^I, x+z, a]"]
    AENC --> CAT
    CAT --> DIT["Wan2.1-VACE-14B<br/>冻结 DiT，仅对 VACE 模块 LoRA 微调<br/>视频与动作潜变量联合去噪（flow matching）"]
    DIT --> GUIDE["修改后的 Inner Guidance 采样<br/>权重 w1、w2"]
    GUIDE --> DROP["丢弃首帧 x^I 与后半段动作潜变量"]
    DROP --> DEC["VAE 解码器 D"]
    DEC --> VID["生成的第一人称人-物操作视频"]
```

图依据：论文 §3.2（Model Design）、§4.1 / Fig.2、Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 093/249 |
| 分组 | 27 Embodied AI & Vision-Language-Action Ego-VLA |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2603.27449> |

## 核心机制（归纳）

### 策展导读要点

An egocentric world model that generates realistic human-object interactions as videos conditioned on an input image, a text prompt, and per-frame human actions, including both body poses and hand gestures

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2603.27449) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **27 Embodied AI & Vision-Language-Action Ego-VLA** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「LOME」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **27 Embodied AI & Vision-Language-Action Ego-VLA**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2603_27449_lome-learning-human-object-manipulation.md`](../../sources/papers/sun_awesome_ego_2603_27449_lome-learning-human-object-manipulation.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2603.27449>
- [arXiv PDF](https://arxiv.org/pdf/2603.27449)：已通读方法部分（实验与附录以略读为准）：§1、§3（Preliminary、Model Design：Action Maps、Joint Action-Environment Modeling、Modified Guidance）、§4.1 实现细节；§4.2 起实验仅略读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2603.27449)
