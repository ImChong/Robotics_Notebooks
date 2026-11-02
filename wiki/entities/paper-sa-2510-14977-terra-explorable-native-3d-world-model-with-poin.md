---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2510.14977"
venue: "arXiv 2025"
summary: "Explorable native 3D world model with point latents."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2510_14977_terra-explorable-native-3d-world-model-w.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# Terra

**Terra: Explorable Native 3D World Model with Point Latents** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 103/571** 篇，分组 **43 3D & 4D World Models General Purpose**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Explorable native 3D world model with point latents.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Explorable native 3D world model with point latents.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["输入：彩色点云 Q（2D 图像经 2D-to-3D 逆投影得到）"] --> ENC["P2G-VAE 编码器（Point Transformer）：下采样为点潜变量 P（位置 + 特征）"]
    ENC --> PER["Robust Position Perturbation：位置加高斯噪声"]
    PER --> DEC["P2G-VAE 解码器：Adaptive Upsample + Adaptive Refine 逐步致密化"]
    DEC --> GS["3D Gaussian primitives"]
    GS --> RND["栅格化渲染，渲染监督 + Chamfer / normal / rank / 显式颜色损失"]
    ENC -->|"点潜变量作为生成建模对象"| SP
    SP["SPFlow：稀疏 3D 卷积 UNet，联合去噪点位置与特征（flow matching）"] --> TS["Distance-Aware Trajectory Smoothing：用 JV 算法匹配噪声与点"]
    CON["条件模块：crop / uniform / 组合 mask 条件点潜变量"] --> SP
    SP --> OUT["生成点潜变量，再经 P2G-VAE 解码器得到 Gaussians"]
    OUT --> EXP["Outpainting：以已知区域为条件逐步扩展，得到可探索场景"]
    EXP -->|"下一步探索"| CON
```

图依据：论文 §3 Proposed Approach（Fig. 2 整体流水线、Fig. 3 方法细节；3.1 Latent Point Representation、3.2 P2G-VAE、3.3 Native 3D Generative Modeling）。2D-to-3D 逆投影来自 Fig. 2 图中标注。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 103/571 |
| 分组 | 43 3D & 4D World Models General Purpose |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2510.14977> |
| 项目页 | <https://huang-yh.github.io/terra/> |

## 核心机制（归纳）

### 策展导读要点

Explorable native 3D world model with point latents.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2510.14977) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **43 3D & 4D World Models General Purpose** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Terra」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2510_14977_terra-explorable-native-3d-world-model-w.md`](../../sources/papers/sun_awesome_wm_2510_14977_terra-explorable-native-3d-world-model-w.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2510.14977>
- [arXiv PDF](https://arxiv.org/pdf/2510.14977)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2 相关工作（略读）、§3 方法（3.1–3.3，含三阶段训练与条件机制）、§4.1 数据集

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2510.14977)
