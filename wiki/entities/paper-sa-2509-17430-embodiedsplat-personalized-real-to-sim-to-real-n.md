---
type: entity
tags: [paper, curated-index, awesome-real2sim2real, sun254667-r2s2r]
status: complete
updated: 2026-10-06
arxiv: "2509.17430"
venue: "CVPR 2025"
summary: "Personalizes policy training by efficiently capturing scenes with 3D Gaussian Splatting, analyzing training strategies, pre-training datasets, and mesh reconstruction techniques"
related:
  - ../entities/awesome-real2sim2real.md
  - ../overview/sun-awesome-r2s2r-technology-map.md
  - ../methods/reinforcement-learning.md
  - ../methods/crisp-real2sim.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/sun_awesome_r2s2r_2509_17430_embodiedsplat-personalized-real-to-sim-t.md
  - ../../sources/papers/sun_awesome_r2s2r_catalog.md
  - ../../sources/repos/awesome-real2sim2real.md
---

# EmbodiedSplat

**EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device** 收录于 [Awesome-Real2Sim2Real](https://github.com/sun254667/Awesome-Real2Sim2Real) **第 045/063** 篇，分组 **42 Simulation Frameworks & Pipelines**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Personalizes policy training by efficiently capturing scenes with 3D Gaussian Splatting, analyzing training strategies, pre-training datasets, and mesh reconstruction techniques

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真策略迁移到真机 |
| Real2Sim | Real to Simulation | 真机数据重建/校准仿真 |
| R2S2R | Real2Sim2Real | 真机→仿真→真机闭环 |
| DR | Domain Randomization | 域随机化 |

## 为什么重要

- Personalizes policy training by efficiently capturing scenes with 3D Gaussian Splatting, analyzing training strategies, pre-training datasets, and mesh reconstruction techniques
- 想横向对照同一分组的其他工作，可以从 [Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  subgraph A["(a) 场景采集"]
    CAP["iPhone + Polycam 录制 RGB-D"] --> NS["Nerfstudio 处理：采样 1000 帧对齐的 RGB、深度与位姿"]
  end
  subgraph B["(b) 重建网格"]
    NS --> DN["DN-Splatter 训练 3D Gaussian Splatting（深度与法线正则，法线来自 Metric3D-V2）"]
    DN --> PM["Poisson 重建得到网格 .ply（DN mesh）"]
  end
  CAP -.Polycam 自带网格，作对比.-> PCM["Polycam mesh"]
  subgraph C["(c) 仿真训练"]
    PM --> BL["Blender 转为 glb，载入 Habitat-Sim"]
    PCM --> BL
    BL --> EP["取最大 navmesh 岛，生成 ImageNav 训练 / 评测 episodes"]
    PRE["HM3D 或 HSSD 上预训练的 ImageNav 策略"] --> FT["在重建场景中微调策略"]
    EP --> FT
  end
  subgraph D["(d) 真实部署"]
    FT --> DEP["在同一真实场景部署，做 image-goal navigation"]
  end
```

图依据：论文 §3 / Fig. 2（四阶段 a–d 流水线）、§3.1–§3.3、§4.2（微调设定）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 045/063 |
| 分组 | 42 Simulation Frameworks & Pipelines |
| 出处 | CVPR 2025 |
| 论文 | <https://arxiv.org/abs/2509.17430> |

## 核心机制（归纳）

### 策展导读要点

Personalizes policy training by efficiently capturing scenes with 3D Gaussian Splatting, analyzing training strategies, pre-training datasets, and mesh reconstruction techniques

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2509.17430) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-r2s2r-technology-map.md) 的 **42 Simulation Frameworks & Pipelines** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EmbodiedSplat」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **42 Simulation Frameworks & Pipelines**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome-Real2Sim2Real](../entities/awesome-real2sim2real.md)
- 技术地图：[Awesome-Real2Sim2Real 技术地图](../overview/sun-awesome-r2s2r-technology-map.md)
- 方法/任务：[reinforcement-learning.md](../methods/reinforcement-learning.md)、[locomotion.md](../tasks/locomotion.md)

## 参考来源

- [`sources/papers/sun_awesome_r2s2r_2509_17430_embodiedsplat-personalized-real-to-sim-t.md`](../../sources/papers/sun_awesome_r2s2r_2509_17430_embodiedsplat-personalized-real-to-sim-t.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_r2s2r_catalog.md`](../../sources/papers/sun_awesome_r2s2r_catalog.md) — 列表总表
- [`sources/repos/awesome-real2sim2real.md`](../../sources/repos/awesome-real2sim2real.md)
- 论文：<https://arxiv.org/abs/2509.17430>
- [arXiv PDF](https://arxiv.org/pdf/2509.17430)：已通读方法部分（§1 引言、§3 方法）；实验（§4）略读：§4.1–§4.2。

## 推荐继续阅读

- [Awesome-Real2Sim2Real 仓库](https://github.com/sun254667/Awesome-Real2Sim2Real)
- [原文](https://arxiv.org/abs/2509.17430)
