---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2608.25956"
venue: "2026"
summary: "These models can achieve exceptional visual quality, but they lack explicit spatial structure for individual objects and repeatedly process redundant background content."
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2608_25956_4dgs-wam-bridging-past-and-future-with-a.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# 4DGS-WAM

**4DGS-WAM: Bridging Past and Future with an Object-Centric World Action Model based on 4D Gaussian Splatting** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 272/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

These models can achieve exceptional visual quality, but they lack explicit spatial structure for individual objects and repeatedly process redundant background content.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- These models can achieve exceptional visual quality, but they lack explicit spatial structure for individual objects and repeatedly process redundant background content.
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["过去 RGB 帧 I_0:t"] --> B["4DGS 重建器<br/>得到 Gaussian 基元 G_0:t，取当前活跃快照 G_t"]
    A --> C["视觉基础模型 VFM<br/>分割 / 光流 / 深度 / 相机位姿"]
    C --> D["物体掩码关联 Gaussians<br/>动态物体 G_t^(o) + 静态背景 G_static"]
    C --> E["抬升到 3D：物体中心轨迹 T_0:t 与 3D 流对应 F"]
    B --> D
    D --> S["抽象状态 s_t"]
    E --> S
    S --> P["时间条件策略 π(s_t, h)<br/>memory token + 因果时间注意力 + actor kNN 注意力<br/>一次评估得到每个 actor 的目标时域 SE(3) 动作"]
    P -->|"含 ego 动作 → 未来相机位姿"| W
    D --> W["世界模型 W(G_t, a, h)<br/>part tokens 上预测残差 SE(3) twist，混合到 splats 并与动作复合；残差外观更新"]
    W --> G["未来 splats Ĝ_t′ ∪ G_static"]
    G --> R["在目标相机位姿渲染未来观测 Î_t′"]
    E -.->|"训练：3D 流监督 W"| W
```

图依据：论文 §3.1–3.2 / Fig.1、Fig.3、Eq.(1)–(5)

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 272/564 |
| 分组 | WAMs |
| 出处 | 2026 |
| 论文 | <https://arxiv.org/abs/2608.25956> |
| 子类 / 象限 | 自动驾驶 · 三维表示与状态估计 · 四象限外 |

## 核心机制（归纳）

### 策展导读要点

These models can achieve exceptional visual quality, but they lack explicit spatial structure for individual objects and repeatedly process redundant background content.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2608.25956) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「4DGS-WAM」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **WAMs**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2608_25956_4dgs-wam-bridging-past-and-future-with-a.md`](../../sources/papers/rcl_awesome_wam_2608_25956_4dgs-wam-bridging-past-and-future-with-a.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2608.25956>
- [arXiv PDF](https://arxiv.org/pdf/2608.25956)：已通读方法部分（实验与附录以略读为准）：§3.1–3.2（含 Fig.3）

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2608.25956)
