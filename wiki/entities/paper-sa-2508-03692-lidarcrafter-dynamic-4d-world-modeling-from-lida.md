---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2508.03692"
venue: "arXiv 2025"
code: https://github.com/lidarcrafter/toolkit
summary: "Dynamic 4D world modeling from LiDAR sequences."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2508_03692_lidarcrafter-dynamic-4d-world-modeling-f.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# LiDARCrafter

**LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 455/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Dynamic 4D world modeling from LiDAR sequences.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Dynamic 4D world modeling from LiDAR sequences.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["用户自由文本指令"] --> B["Text2Layout：LLM 构建以自车为中心的场景图 G=(V,E)（节点含类别与运动状态，边为空间关系）"]
    B --> C["图融合编码器：冻结 CLIP 文本嵌入 + L 层 TripletGCN"]
    C --> D["三分支扩散解码器：3D 框 b、位移轨迹 δ、形状先验点 p"]
    D --> E["4D 前景布局 Oi"]
    B --> F["Layout2Scene：稀疏物体条件嵌入（位置、类别、框）"]
    E --> F
    F --> G["距离图（range image）扩散 U-Net 去噪，生成首帧静态 LiDAR 扫描 I0"]
    G --> H["Scene2Seq：反投影为点云，按布局框拆分背景与前景"]
    H --> I["背景按自车位姿变换，前景按各自运动先验变换（warp）"]
    I --> J["球面投影得条件距离图 Icond，与噪声样本拼接送入扩散主干"]
    J --> K["自回归生成下一帧距离图，逐帧迭代"]
    K --> I
    K --> L["时序连贯的 4D LiDAR 序列"]
    G -.-> M["布局驱动场景编辑：修改布局后掩码混合重新去噪"]
    L --> N["EvalSuite：物体、布局、时序三类指标评测"]
```

图依据：论文 §3（三阶段框架）、§3.1 Text2Layout、§3.2 Layout2Scene、§3.3 Scene2Seq、§3.4 EvalSuite / Fig.2–4。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 455/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2508.03692> |
| 代码/项目 | <https://github.com/lidarcrafter/toolkit> |

## 核心机制（归纳）

### 策展导读要点

Dynamic 4D world modeling from LiDAR sequences.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2508.03692) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「LiDARCrafter」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **82 Autonomous Driving**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2508_03692_lidarcrafter-dynamic-4d-world-modeling-f.md`](../../sources/papers/sun_awesome_wm_2508_03692_lidarcrafter-dynamic-4d-world-modeling-f.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2508.03692>
- [arXiv PDF](https://arxiv.org/pdf/2508.03692)：已通读方法部分（实验与附录以略读为准）：§3 LiDARCrafter 全部小节（3.1–3.4）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2508.03692)
