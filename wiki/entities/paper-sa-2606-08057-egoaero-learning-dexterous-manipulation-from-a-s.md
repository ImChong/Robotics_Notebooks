---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2606.08057"
venue: "arXiv 2026"
summary: "A dexterous manipulation learning framework without pre-scanned object assets, constructing the large-scale EgoDex-R dataset (4.3M RGB-D frames)"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2606_08057_egoaero-learning-dexterous-manipulation.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# EgoAERO

**EgoAERO: Learning Dexterous Manipulation from a Single Egocentric Video without Object Assets** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 012/249** 篇，分组 **22 Hand-Object Interaction & Fine-grained Activity**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A dexterous manipulation learning framework without pre-scanned object assets, constructing the large-scale EgoDex-R dataset (4.3M RGB-D frames)

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- A dexterous manipulation learning framework without pre-scanned object assets, constructing the large-scale EgoDex-R dataset (4.3M RGB-D frames)
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["单段第一人称 RGB-D 人类示范 + 任务描述"] --> B["MLLM 语义预处理：识别目标物体，生成 SAM3 提示，选种子帧"]
    B --> C["SAM3 逐帧物体 mask"]
    C --> D["无资产物体跟踪：RANSAC 粗位姿 → keyframe memory-pool 位姿优化"]
    D --> E["神经物体场粗网格 + SAM3D 细节网格，刚性与尺度对齐"]
    A --> F["HaWoR 的 MANO 手部估计 + RGB-D 深度修正全局平移"]
    A --> G["RGB-D SLAM 估计相机轨迹，统一到固定桌面坐标系（Ego 运动补偿）"]
    D --> H["Adaptive Contact Optimization：有界手部平移与局部指尖修正"]
    E --> H
    F --> H
    G --> H
    H --> I["接触一致的手-物体轨迹"]
    I --> J["Stage I：手部轨迹跟踪策略（运动学重定向热启动）"]
    J --> K["Stage II：物体-接触残差策略，a = a_I + Δa_R"]
    K --> L["仿真灵巧手策略，执行到真实机器人（Fig.1）"]
    I -.-> M["在线质量评估：accept / repairable accept / recapture"]
```

图依据：论文 §2.1（Fig.2 重建流程）、§2.2 两阶段残差策略、§3 在线质量评估 / Fig.1

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 012/249 |
| 分组 | 22 Hand-Object Interaction & Fine-grained Activity |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.08057> |

## 核心机制（归纳）

### 策展导读要点

A dexterous manipulation learning framework without pre-scanned object assets, constructing the large-scale EgoDex-R dataset (4.3M RGB-D frames)

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.08057) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **22 Hand-Object Interaction & Fine-grained Activity** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EgoAERO」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_ego_2606_08057_egoaero-learning-dexterous-manipulation.md`](../../sources/papers/sun_awesome_ego_2606_08057_egoaero-learning-dexterous-manipulation.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2606.08057>
- [arXiv PDF](https://arxiv.org/pdf/2606.08057)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2 方法（2.1 无资产手物重建、2.2 两阶段残差策略）、§3 数据采集与质量评估；附录 A–E 未读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2606.08057)
