---
type: entity
tags: [paper, curated-index, awesome-world-models, sun254667-wm]
status: complete
updated: 2026-10-06
arxiv: "2312.17655"
venue: "arXiv 2023"
code: https://github.com/OpenDriveLab/ViDAR
summary: "Visual point cloud forecasting enables scalable autonomous driving."
related:
  - ../entities/awesome-world-models.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/sun_awesome_wm_2312_17655_vidar-visual-point-cloud-forecasting-ena.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
  - ../../sources/repos/awesome-world-models.md
---

# ViDAR

**ViDAR: Visual Point Cloud Forecasting enables Scalable Autonomous Driving** 收录于 [Awesome World Models](https://github.com/sun254667/awesome-world-models) **第 493/571** 篇，分组 **82 Autonomous Driving**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Visual point cloud forecasting enables scalable autonomous driving.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| MBRL | Model-Based RL | 基于模型的强化学习 |

## 为什么重要

- Visual point cloud forecasting enables scalable autonomous driving.
- 想横向对照同一分组的其他工作，可以从 [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World Models](../entities/awesome-world-models.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart LR
  IMG["多视角历史图像序列 I"] --> HE["History Encoder\n视觉 BEV 编码器 (预训练目标)\n输出 BEV 特征 F_bev"]
  HE --> LR["Latent Rendering 算子\n在隐空间模拟体渲染，得到几何特征 F^_bev\n多组 (multi-group) 并行"]
  LR --> FD["Future Decoder\n自回归预测未来 BEV 特征 F^_t\n条件为自车运动 e_t 与上一帧 BEV 特征"]
  EGO["期望的自车运动 e_t"] --> FD
  FD --> PROJ["投影层\n生成占据体 P_t"]
  PROJ --> RAY["可微射线投射渲染未来点云"]
  GT["真值未来点云"] --> LOSS["逐射线交叉熵监督"]
  RAY --> LOSS
  HE -->|"预训练后迁移"| DOWN["下游: BEVFormer 3D 检测 / OccNet 语义占据 / UniAD 感知-预测-规划"]
```

图依据：论文 §3.1 Overview（Fig. 2）、§3.2 Latent Rendering（Fig. 3 / Fig. 4）、§3.3 Future Decoder（Fig. 5）；下游验证见 §4 Implementation Details。逐射线交叉熵监督的具体连接依据论文 Loss 一段。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 493/571 |
| 分组 | 82 Autonomous Driving |
| 出处 | arXiv 2023 |
| 论文 | <https://arxiv.org/abs/2312.17655> |
| 代码/项目 | <https://github.com/OpenDriveLab/ViDAR> |

## 核心机制（归纳）

### 策展导读要点

Visual point cloud forecasting enables scalable autonomous driving.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2312.17655) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-wm-technology-map.md) 的 **82 Autonomous Driving** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「ViDAR」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_wm_2312_17655_vidar-visual-point-cloud-forecasting-ena.md`](../../sources/papers/sun_awesome_wm_2312_17655_vidar-visual-point-cloud-forecasting-ena.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_wm_catalog.md`](../../sources/papers/sun_awesome_wm_catalog.md) — 列表总表
- [`sources/repos/awesome-world-models.md`](../../sources/repos/awesome-world-models.md)
- 论文：<https://arxiv.org/abs/2312.17655>
- [arXiv PDF](https://arxiv.org/pdf/2312.17655)：已通读方法部分（实验与附录以略读为准）：§3 Methodology（3.1–3.3）与 §4 的 Implementation Details / Loss 小节

## 推荐继续阅读

- [Awesome World Models 仓库](https://github.com/sun254667/awesome-world-models)
- [原文](https://arxiv.org/abs/2312.17655)
