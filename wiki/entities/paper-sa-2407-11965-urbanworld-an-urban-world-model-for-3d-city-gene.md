---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2407.11965"
venue: "arXiv 2024"
summary: "Urban world model for 3D city generation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2407_11965_urbanworld-an-urban-world-model-for-3d-c.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# UrbanWorld

**UrbanWorld: An Urban World Model for 3D City Generation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 104/571** 篇，分组 **43 3D & 4D World Models General Purpose**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Urban world model for 3D city generation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Urban world model for 3D city generation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    LAYOUT["2.5D 城市布局数据：OSM / 语义+高度图"] --> MESH["3D 城市布局生成：无纹理 3D 网格，资产逐个拆分并记录中心位置"]
    INSTR["用户文本指令"] --> MLLM["Urban MLLM（VILA-1.5 微调）：为每个资产生成外观与材质描述"]
    MESH --> REND
    MLLM --> REND
    REF["可选：参考图像"] --> REND
    subgraph REND["逐资产纹理渲染（Fig.2）"]
        DEPTH["多视角深度图"] --> CN["深度感知 ControlNet + 2D 扩散：生成多视角图像"]
        CN --> UV["反投影到 UV 空间并按视角掩码合并"]
        UV --> INP["UV 位置图编码器引导的纹理补全（inpainting）"]
        INP --> UP["ControlNet-tile 图像放大"]
    end
    REND --> ASM["按布局元数据位置重组资产"]
    ASM --> REF2["MLLM 辅助精修：Urban MLLM 找出与提示不一致处并给出修改后的提示"]
    REF2 -->|"重新渲染相关资产"| REND
    REF2 --> OUT["细节更高的 3D 城市环境"]
```

图依据：论文 §3 Methodology（§3.1 布局生成、§3.2 MLLM 场景设计、§3.3 纹理渲染 Fig.2、§3.4 精修 Fig.3）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 104/571 |
| 分组 | 43 3D & 4D World Models General Purpose |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2407.11965> |

## 核心机制（归纳）

### 策展导读要点

Urban world model for 3D city generation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2407.11965) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **43 3D & 4D World Models General Purpose** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「UrbanWorld」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **43 3D & 4D World Models General Purpose**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2407_11965_urbanworld-an-urban-world-model-for-3d-c.md`](../../sources/papers/sun_awesome_wm_2407_11965_urbanworld-an-urban-world-model-for-3d-c.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2407.11965>
- [arXiv PDF](https://arxiv.org/pdf/2407.11965)：已通读方法部分（实验与附录以略读为准）：§3 方法（布局生成、MLLM 场景设计、纹理渲染、MLLM 精修）；§4 实验略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2407.11965)
