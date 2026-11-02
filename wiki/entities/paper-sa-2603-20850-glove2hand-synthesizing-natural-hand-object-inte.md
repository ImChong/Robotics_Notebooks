---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2603.20850"
venue: "CVPR 2026 (Highlight)"
summary: "Generates realistic bare-hand HOI videos from multi-modal sensing glove data, accompanied by the HandSense dataset"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2603_20850_glove2hand-synthesizing-natural-hand-obj.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Glove2Hand

**Glove2Hand: Synthesizing Natural Hand-Object Interaction from Multi-Modal Sensing Gloves** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 020/249** 篇，分组 **22 Hand-Object Interaction & Fine-grained Activity**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Generates realistic bare-hand HOI videos from multi-modal sensing glove data, accompanied by the HandSense dataset

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Generates realistic bare-hand HOI videos from multi-modal sensing glove data, accompanied by the HandSense dataset
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TB
  subgraph TR["训练"]
    direction LR
    A1["裸手视频 + 手部姿态 P"] --> A2["3D Gaussian 手模型\n渲染 hand-only 帧"]
    A2 --> A3["叠加到原视频\n并遮蔽腕部区域"]
    A3 --> A4["编码为潜变量，作为条件"]
    A4 --> A5["Diffusion hand restorer\nControlNet + AnimateDiff\n恢复原始裸手视频"]
  end
  subgraph INF["推理"]
    direction LR
    B1["手套视频 + 手套姿态 P"] --> B2["Grounding DINO + SAM-2\n提取手套 / 物体掩码"]
    B2 --> B3["Propainter 光流修复\n擦除手套区域，保留物体像素"]
    B1 --> B4["3D Gaussian 手模型\n渲染 hand-only 帧"]
    B3 --> B5["粘贴渲染手"]
    B4 --> B5
    B5 --> B6["Diffusion hand restorer"]
    B6 --> B7["逼真裸手视频"]
  end
  B7 -.->|"用于构建"| DS["HandSense 数据集\n同步触觉 + IMU"]
```

图依据：论文 §3.1–3.2 / Fig.2、Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 020/249 |
| 分组 | 22 Hand-Object Interaction & Fine-grained Activity |
| 出处 | CVPR 2026 (Highlight) |
| 论文 | <https://arxiv.org/abs/2603.20850> |

## 核心机制（归纳）

### 策展导读要点

Generates realistic bare-hand HOI videos from multi-modal sensing glove data, accompanied by the HandSense dataset

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2603.20850) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **22 Hand-Object Interaction & Fine-grained Activity** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Glove2Hand」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **22 Hand-Object Interaction & Fine-grained Activity**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2603_20850_glove2hand-synthesizing-natural-hand-obj.md`](../../sources/papers/sun_awesome_ego_2603_20850_glove2hand-synthesizing-natural-hand-obj.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2603.20850>
- [arXiv PDF](https://arxiv.org/pdf/2603.20850)：已通读方法部分（实验与附录以略读为准）：Abstract、§1、§3 方法（3.1–3.2）、Fig.2–3 图注

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2603.20850)
