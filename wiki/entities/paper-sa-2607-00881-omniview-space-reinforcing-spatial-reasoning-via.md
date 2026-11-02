---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2607.00881"
venue: "arXiv 2026"
summary: "A framework maintaining spatial consistency through multimodal egocentric evidence with three core components: Multi-Perspective Spatial Mapping, Tool-Guided Egocentric Reasoning, and Cognitive-Map Distillation"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2607_00881_omniview-space-reinforcing-spatial-reaso.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# OmniView-Space

**OmniView-Space: Reinforcing Spatial Reasoning via Multi-Perspective Spatial Mapping** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 190/249** 篇，分组 **Advanced Reasoning & Cutting-edge Advances**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A framework maintaining spatial consistency through multimodal egocentric evidence with three core components: Multi-Perspective Spatial Mapping, Tool-Guided Egocentric Reasoning, and Cognitive-Map Distillation

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- A framework maintaining spatial consistency through multimodal egocentric evidence with three core components: Multi-Perspective Spatial Mapping, Tool-Guided Egocentric Reasoning, and Cognitive-Map Distillation
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["问题 q + 多视角图像 I"] --> POL["策略 MLLM 分析问题<br/>选择 ego anchor（相机 / 物体 / 方向）与 object hints"]
    subgraph MPSM["MPSM 工具包（Sec. 3.1）"]
        R1["Step 1：场景重建<br/>稠密几何模型预测 point map / 置信度 / 位姿<br/>promptable 分割器接地 object hints<br/>置信度门控、融合、体素化、聚类成物体实例"]
        R2["Step 2：ego 系实例化<br/>重力对齐后按 anchor 取原点与朝向<br/>变换到 query 指定 ego 系"]
        R1 --> R2
        R2 --> VIS["视觉 BEV cognitive map"]
        R2 --> TXT["文本 spatial graph"]
    end
    POL -- "tool_call：cogmap_visualization / cogmap_graph" --> R1
    VIS --> LOOP["返回证据，多轮交织推理"]
    TXT --> LOOP
    LOOP --> POL
    LOOP --> ANS["最终答案"]
    ANS --> RL1["GRPO：答案奖励 + 格式奖励"]
    subgraph DIST["蒸馏（Sec. 3.3）"]
        SFT["MPSM 轨迹 + Qwen3-VL-235B 推理痕迹<br/>构造 SFT 轨迹并过滤"] --> SELF["模型自行预测 JSON 格式 cogmap<br/>渲染为 BEV 后再推理作答"]
        SELF --> RL2["GRPO：答案 + 格式 + map 对齐奖励<br/>MPSM 作参考，仅对答对样本<br/>物体对齐 + 相机对齐"]
    end
    RL1 -.蒸馏.-> SFT
```

图依据：论文 §3 Methodology（§3.1 MPSM / Fig.3、§3.2 / Fig.2、§3.3）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 190/249 |
| 分组 | Advanced Reasoning & Cutting-edge Advances |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2607.00881> |

## 核心机制（归纳）

### 策展导读要点

A framework maintaining spatial consistency through multimodal egocentric evidence with three core components: Multi-Perspective Spatial Mapping, Tool-Guided Egocentric Reasoning, and Cognitive-Map Distillation

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2607.00881) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Advanced Reasoning & Cutting-edge Advances** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「OmniView-Space」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Advanced Reasoning & Cutting-edge Advances**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2607_00881_omniview-space-reinforcing-spatial-reaso.md`](../../sources/papers/sun_awesome_ego_2607_00881_omniview-space-reinforcing-spatial-reaso.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2607.00881>
- [arXiv PDF](https://arxiv.org/pdf/2607.00881)：已通读方法部分（实验与附录以略读为准）：§1、§3.1–3.3（实验 §4 以略读为准）

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2607.00881)
