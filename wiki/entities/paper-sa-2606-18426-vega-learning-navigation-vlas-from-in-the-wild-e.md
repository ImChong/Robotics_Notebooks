---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2606.18426"
venue: "arXiv 2026"
summary: "An approach for training navigation Vision-Language-Action models from unlabeled egocentric navigation videos. Reconstructs local scene geometry, samples navigation goals, and generates obstacle-aware trajectories. Reduc"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2606_18426_vega-learning-navigation-vlas-from-in-th.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# VEGA

**VEGA: Learning Navigation VLAs from In-the-Wild Egocentric Video with Geometric Trajectory Supervision** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 100/249** 篇，分组 **27 Embodied AI & Vision-Language-Action Ego-VLA**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

An approach for training navigation Vision-Language-Action models from unlabeled egocentric navigation videos. Reconstructs local scene geometry, samples navigation goals, and generates obstacle-aware trajectories. Reduces collisions by 33.0% and improves obstacle clearance by 17.9%

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- An approach for training navigation Vision-Language-Action models from unlabeled egocentric navigation videos. Reconstructs local scene geometry, samples navigation goals, and generates obstacle-aware trajectories. Reduces collisions by 33.0% and improves obstacle clearance by 17.9%
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["无标注第一人称导航视频<br/>按固定帧率采样 RGB 帧 I_t"] --> B["MoGe-2 重建稠密度量点图 P_t"]
    B --> C["RANSAC 拟合地面<br/>对齐到机器人坐标系"]
    C --> D["投影到局部 BEV<br/>高度过滤得障碍占用栅格"]
    D --> E["光线投射得可见性掩码<br/>区分已知自由 / 占用 / 未观测"]
    E --> F["计算 Visibility-aware ESDF"]
    A --> G["Florence2 检测物体标签与框"]
    G --> H["映射到点图并投影到地面<br/>得文本 / 图像区域 / 路点三种目标"]
    F --> I["自由空间中额外采样路点目标"]
    H --> J["航向感知 Dijkstra 搜索<br/>在 ESDF 上生成无碰撞路径"]
    I --> J
    J --> K["ESDF 上优化内部路点<br/>避障 + 平滑，得参考轨迹"]
    K --> L["训练元组<br/>I_t + 目标 + 目标模态 + 参考轨迹"]
    L --> M["基于 pi0.5 的多模态目标条件 VLA<br/>SigLIP / Gemma 冻结，路点编码器与动作专家训练"]
    M --> N["flow matching 目标训练"]
    N --> O["部署：仅 RGB 与目标<br/>动作专家输出路点轨迹，无需重建或 ESDF"]
```

图依据：论文 §IV-A、§IV-B / Fig.2、Fig.3

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 100/249 |
| 分组 | 27 Embodied AI & Vision-Language-Action Ego-VLA |
| 出处 | arXiv 2026 |
| 论文 | <https://arxiv.org/abs/2606.18426> |

## 核心机制（归纳）

### 策展导读要点

An approach for training navigation Vision-Language-Action models from unlabeled egocentric navigation videos. Reconstructs local scene geometry, samples navigation goals, and generates obstacle-aware trajectories. Reduces collisions by 33.0% and improves obstacle clearance by 17.9%

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2606.18426) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **27 Embodied AI & Vision-Language-Action Ego-VLA** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「VEGA」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_ego_2606_18426_vega-learning-navigation-vlas-from-in-th.md`](../../sources/papers/sun_awesome_ego_2606_18426_vega-learning-navigation-vlas-from-in-th.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2606.18426>
- [arXiv PDF](https://arxiv.org/pdf/2606.18426)：已通读方法部分（实验与附录以略读为准）：§III 预备知识、§IV 方法（A 轨迹分布生成、B 多模态目标条件 VLA）、§V-A 实验设置开头

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2606.18426)
