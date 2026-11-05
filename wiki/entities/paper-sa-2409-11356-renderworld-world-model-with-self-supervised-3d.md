---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2409.11356"
venue: "arXiv 2024"
summary: "World model with self-supervised 3D labels."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2409_11356_renderworld-world-model-with-self-superv.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# RenderWorld

**RenderWorld: World Model with Self-Supervised 3D Label** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 478/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

World model with self-supervised 3D labels.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- World model with self-supervised 3D labels.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["多相机图像 Img_i<br/>+ 内外参"] --> B["Img2Occ：BEVStereo4D + Swin 2D 特征<br/>→ 插值到 3D 体特征"]
    B --> C["体素中心初始化带语义标签的 anchor<br/>→ 3D Gaussian"]
    C --> D["Gaussian Splatting 渲染<br/>多相机 2D 语义图 + 深度图"]
    E["LiDAR 投影得到的<br/>2D 语义 / 深度真值"] --> F["L_sem 交叉熵 + L_dep Pearson 相关"]
    D --> F
    C --> G["训练完成后生成 3D occupancy 标签"]
    G --> H["AM-VAE：拆分 air / non-air 体素"]
    H --> I["两个 VQVAE（各自码本）<br/>离散 token s_Air / s_N-Air"]
    I --> J["世界模型：场景 token T + 车辆 token z0<br/>分层 token + masked 时间注意力"]
    J --> K["预测下一时刻 token"]
    K --> L["场景解码器 d → 预测 occupancy"]
    K --> M["车辆解码器 d_ego → 自车位移 / 规划轨迹"]
```

图依据：论文 §III-A / Fig.2（Img2Occ）、§III-B（AM-VAE）、§III-C（世界模型）、Fig.1（总体流程）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 478/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2409.11356> |

## 核心机制（归纳）

### 策展导读要点

World model with self-supervised 3D labels.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2409.11356) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「RenderWorld」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2409_11356_renderworld-world-model-with-self-superv.md`](../../sources/papers/sun_awesome_wm_2409_11356_renderworld-world-model-with-self-superv.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2409.11356>
- [arXiv PDF](https://arxiv.org/pdf/2409.11356)：已通读方法部分（实验与附录以略读为准）：§III 方法（III-A Img2Occ、III-B AM-VAE、III-C 世界模型）、§IV-A 实验设定

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2409.11356)
