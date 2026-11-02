---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2606.15200"
venue: "ICML 2026"
summary: "Introduces UCS-Bench, a dataset spanning 170+ hours with 8.1K+ timestamped questions for User-Centric Continual Spatial intelligence"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2606_15200_keep-it-in-mind-user-centric-continual-s.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Keep It in Mind

**Keep It in Mind: User Centric Continual Spatial Intelligence Reasoning in Egocentric Video Streams** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 043/249** 篇，分组 **24 Long-form Video Understanding & Structured Reasoning**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Introduces UCS-Bench, a dataset spanning 170+ hours with 8.1K+ timestamped questions for User-Centric Continual Spatial intelligence

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Introduces UCS-Bench, a dataset spanning 170+ hours with 8.1K+ timestamped questions for User-Centric Continual Spatial intelligence
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph BG["后台：视频到结构化记忆（与提问无关）"]
        IN["egocentric 视频流，1 fps"]
        GEO["DepthAnything3<br/>逐帧度量深度 + 相机位姿 T_w←e"]
        DET["GroundingDINO (Objects365 词表) 检测<br/>→ SAM2 传播 mask，得到持久实例轨迹"]
        TAG["轻量 MLLM 场景描述器<br/>逐帧语义场景标签"]
        MAP["世界对齐度量-语义地图 M<br/>以第一帧初始化并增量更新"]
        SG["演化场景图 G=(V,E)<br/>物体节点 + 场景区域节点；物体-物体 / 物体-区域边"]
        IN --> GEO
        IN --> DET
        IN --> TAG
        GEO --> MAP
        DET --> MAP
        TAG --> MAP
        MAP --> SG
    end
    subgraph ON["在线：问题时刻 t 的问答"]
        Q["问题 → 解析为结构化查询意图"]
        SUB["仅取 t 之前观测到的节点<br/>按意图打分，保留 top-K 实例 + 关键帧（最小子图）"]
        EGO["用 T_e←w = T_w←e^-1 把世界坐标转为当前 ego 坐标<br/>得到相对方向 / 距离 / 可达性边"]
        LLM["MLLM 以子图 + 关键帧 + 问题生成回答"]
        Q --> SUB --> EGO --> LLM
    end
    SG --> SUB
```

图依据：论文 §4 / Fig.5（§4.1 概述、§4.2 离线空间场景图记忆、§4.3 在线视频推理）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 043/249 |
| 分组 | 24 Long-form Video Understanding & Structured Reasoning |
| 出处 | ICML 2026 |
| 论文 | <https://arxiv.org/abs/2606.15200> |

## 核心机制（归纳）

### 策展导读要点

Introduces UCS-Bench, a dataset spanning 170+ hours with 8.1K+ timestamped questions for User-Centric Continual Spatial intelligence

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.15200) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **24 Long-form Video Understanding & Structured Reasoning** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Keep It in Mind」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **24 Long-form Video Understanding & Structured Reasoning**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2606_15200_keep-it-in-mind-user-centric-continual-s.md`](../../sources/papers/sun_awesome_ego_2606_15200_keep-it-in-mind-user-centric-continual-s.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2606.15200>
- [arXiv PDF](https://arxiv.org/pdf/2606.15200)：已通读方法部分（实验与附录以略读为准）：§4 方法（DirectMe 概述、离线空间场景图记忆、在线视频推理）；§3 UCS-Bench 构建、§5 实验与附录略读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2606.15200)
