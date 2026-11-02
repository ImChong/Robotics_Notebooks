---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2503.18945"
venue: "arXiv 2025"
summary: "Geometric-aware unified world modeling."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2503_18945_aether-geometric-aware-unified-world-mod.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Aether

**Aether: Geometric-Aware Unified World Modeling** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 091/571** 篇，分组 **43 3D & 4D World Models General Purpose**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Geometric-aware unified world modeling.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Geometric-aware unified world modeling.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  SYN["合成 RGB-D 视频"] --> DM["动态掩码：Grounded SAM 2 分割潜在动态物体"]
  DM --> SL["视频切片：剔除 SIFT 点不足、动态区域过大、大运动或光流不一致的帧"]
  SL --> CE["粗相机估计：DroidCalib"]
  CE --> CR["相机精修：CoTracker3 跟踪静态区域特征点 + 光束法平差"]
  CR --> ANN["带相机内外参标注的 RGB-D 视频"]
  ANN --> DEP["深度视频 → 视差 → 尺度不变归一化 → 3D VAE → 深度 latent"]
  ANN --> RAY["相机轨迹 → raymap（6 通道）→ 下采样对齐 → 动作 latent"]
  ANN --> COL["彩色视频 → 3D VAE → 颜色 latent"]
  DEP --> CAT["通道拼接；条件随机掩码"]
  RAY --> CAT
  COL --> CAT
  CAT --> DIF["CogVideoX-5b-I2V 初始化的视频扩散模型：阶段 1 latent MSE，阶段 2 解码后加 MS-SSIM、深度与点图损失"]
  DIF --> T1["4D 重建：条件为完整彩色视频"]
  DIF --> T2["视频预测：条件为首帧，可选相机轨迹"]
  DIF --> T3["目标条件视觉规划：条件为首帧与目标帧，可选相机轨迹"]
```

图依据：论文 §2 / Fig.3（数据标注管线）、§3.1–§3.4 / Fig.4。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 091/571 |
| 分组 | 43 3D & 4D World Models General Purpose |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2503.18945> |
| 项目页 | <https://aether-world.github.io/> |

## 核心机制（归纳）

### 策展导读要点

Geometric-aware unified world modeling.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2503.18945) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **43 3D & 4D World Models General Purpose** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Aether」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2503_18945_aether-geometric-aware-unified-world-mod.md`](../../sources/papers/sun_awesome_wm_2503_18945_aether-geometric-aware-unified-world-mod.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2503.18945>
- [arXiv PDF](https://arxiv.org/pdf/18945)：已通读方法部分（实验与附录以略读为准）：§1、§2（4D 合成数据标注管线）、§3（3.1–3.4）、§5（略读）

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2503.18945)
