---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2602.11021"
venue: "arXiv 2024"
summary: "A differentiable physics‑grounded rigid‑body world model that learns physical laws from sparse videos, using a unified Gaussian representation for visual appearance and collision geometry."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2602_11021_contactgaussian-wm-learning-physics-grou.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# ContactGaussian-WM

**ContactGaussian-WM: Learning Physics-Grounded World Model from Videos** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 039/571** 篇，分组 **114 Physics Simulation & Engine Integration**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

A differentiable physics‑grounded rigid‑body world model that learns physical laws from sparse videos, using a unified Gaussian representation for visual appearance and collision geometry.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- A differentiable physics‑grounded rigid‑body world model that learns physical laws from sparse videos, using a unified Gaussian representation for visual appearance and collision geometry.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    V["接触丰富场景的 RGB 视频（单/多相机，已标定）"] --> SAM["SAM2 提取几何特征"]
    subgraph S1["阶段一：SG-GS 场景初始化"]
        G1["球形各向同性 Gaussian（旋转冻结）"] --> G2["先优化几何属性 G_geo（外观冻结）"]
        G2 --> G3["再优化外观属性 G_vis（几何冻结）"]
    end
    SAM --> G1
    V --> G1
    G3 --> GS["统一的球形 Gaussian 表示 G*"]
    subgraph S2["阶段二：Phys-Geo 精修"]
        X["当前物理状态 + 动作"] --> C["可微碰撞检测（球体并集 SDF，LSE 平滑）"]
        C --> D["无互补性接触动力学<br/>参数 θ = 质量惯量、摩擦、刚度、阻尼"]
        D --> N["下一物理状态"]
        N --> R["前向运动学变换 Gaussian + 3DGS 渲染下一帧图像"]
        N -.->|"滚动到下一时间步"| X
    end
    GS --> C
    GS --> R
    R --> L["图像损失 L_Loft + L_L1"]
    L -.->|"端到端反传：优化 θ，经碰撞交互微调 G_geo"| D
```

图依据：论文 §III A–D / Fig.2

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 039/571 |
| 分组 | 114 Physics Simulation & Engine Integration |
| 出处 | arXiv 2024 |
| 论文 | <https://arxiv.org/abs/2602.11021> |

## 核心机制（归纳）

### 策展导读要点

A differentiable physics‑grounded rigid‑body world model that learns physical laws from sparse videos, using a unified Gaussian representation for visual appearance and collision geometry.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2602.11021) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **114 Physics Simulation & Engine Integration** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ContactGaussian-WM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **114 Physics Simulation & Engine Integration**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_wm_2602_11021_contactgaussian-wm-learning-physics-grou.md`](../../sources/papers/sun_awesome_wm_2602_11021_contactgaussian-wm-learning-physics-grou.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2602.11021>
- [arXiv PDF](https://arxiv.org/pdf/2602.11021)：已通读方法部分（实验与附录以略读为准）：Abstract、§I、§III A–D（问题建模、SG-GS 初始化、Phys-Geo 精修、端到端梯度流）、§IV 实验概述

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2602.11021)
