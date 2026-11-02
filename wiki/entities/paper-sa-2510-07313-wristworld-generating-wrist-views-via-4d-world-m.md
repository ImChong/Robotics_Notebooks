---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2510.07313"
venue: "arXiv 2025"
code: https://github.com/XuWuLingYu/WristWorld
summary: "Generates wrist-views via 4D world models for robotic manipulation."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2510_07313_wristworld-generating-wrist-views-via-4d.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# WristWorld

**WristWorld: Generating Wrist-Views via 4D World Models for Robotic Manipulation** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 344/571** 篇，分组 **812 Manipulation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Generates wrist-views via 4D world models for robotic manipulation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Generates wrist-views via 4D world models for robotic manipulation.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["输入：多路外部（anchor）视角视频"] --> B["VGGT 聚合器：融合多视角特征，重建 4D 点云"]
    B --> C["Wrist Head：可学习 wrist queries 经 Transformer decoder，回归腕部相机外参 R_w、T_w"]
    SPC["SPC 损失：锚点-腕部 2D 匹配，提升为 3D-2D 对，按预测腕部位姿投影，仅用 RGB 监督"] -.-> C
    C --> D["腕视角投影：把重建点云投影到腕部成像平面，得到逐帧对齐的条件图"]
    B --> D
    D --> E["视频 VAE 编码条件图为 z^c，与带噪腕部 latent z^w 沿通道拼接，patch embedding 输入扩展为 2C 通道"]
    A --> F["CLIP 编码各外部视角帧，加文本嵌入、时间嵌入与视角嵌入，形成语义条件 c"]
    E --> G["DiT 视频生成模型"]
    F --> G
    G --> H["生成腕部视角视频，可补充 VLA 训练数据"]
```

图依据：论文 §3 / Fig.2 / Fig.3（§3.2 重建阶段、§3.3 生成阶段）；VLA 数据增强用途取自 Fig.2 图注与 §4.3 标题。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 344/571 |
| 分组 | 812 Manipulation |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2510.07313> |
| 代码/项目 | <https://github.com/XuWuLingYu/WristWorld> |

## 核心机制（归纳）

### 策展导读要点

Generates wrist-views via 4D world models for robotic manipulation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2510.07313) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **812 Manipulation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「WristWorld」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **812 Manipulation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2510_07313_wristworld-generating-wrist-views-via-4d.md`](../../sources/papers/sun_awesome_wm_2510_07313_wristworld-generating-wrist-views-via-4d.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2510.07313>
- [arXiv PDF](https://arxiv.org/pdf/2510.07313)：已通读方法部分（实验与附录以略读为准）：§3.1–§3.3（含 Fig.2、Fig.3）；§4 实验仅略读

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2510.07313)
