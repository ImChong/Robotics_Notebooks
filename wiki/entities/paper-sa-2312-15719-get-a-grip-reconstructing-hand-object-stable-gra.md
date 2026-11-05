---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2312.15719"
venue: "arXiv 2023"
summary: "Reconstructs stable hand-object grasping states from monocular egocentric videos for the first time"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2312_15719_get-a-grip-reconstructing-hand-object-st.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Get a Grip

**Get a Grip: Reconstructing Hand-Object Stable Grasps in Egocentric Videos** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 019/249** 篇，分组 **22 Hand-Object Interaction & Fine-grained Activity**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Reconstructs stable hand-object grasping states from monocular egocentric videos for the first time

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Reconstructs stable hand-object grasping states from monocular egocentric videos for the first time
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  VID["第一视角视频中的稳定抓握片段\n(N 帧 RGB)"] --> HAMER["HaMeR 逐帧重建手\n输出 MANO 手指关节 θ_n 与手到相机位姿 T_h2c"]
  VID --> MASK["物体 2D 分割掩码 M_o\n作为监督"]
  CAD["已知类别的物体 CAD 模型"] --> INIT["初始化\n旋转轴 φ 取 CAD 的 z 轴，各帧转角 ω_n = 0"]
  INIT --> ONEDOF["1-DoF 物体位姿\n全局 T_o2h 后叠加绕轴 φ 的逐帧旋转 ω_n"]
  HAMER -->|"手参数固定"| PROJ["可微投影物体网格\n与手位姿一起渲染到各帧"]
  ONEDOF --> PROJ
  PROJ --> E["联合优化所有帧\nE_mask 掩码 + E_push 穿透 + E_pull 接触吸引"]
  MASK --> E
  E -->|"更新 φ, ω_n, T_o2h, 物体尺度 s"| ONEDOF
  E --> OUT["稳定抓握内的手-物体 3D 重建"]
```

图依据：论文 §3.2 稳定抓握与 1-DoF 假设（Fig. 3 / Fig. 4）、§3.3 Reconstructing Object Poses in a Stable Grasp（Fig. 5，目标函数 E_mask / E_push / E_pull）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 019/249 |
| 分组 | 22 Hand-Object Interaction & Fine-grained Activity |
| 出处 | arXiv 2023 |
| 论文 | <https://arxiv.org/abs/2312.15719> |

## 核心机制（归纳）

### 策展导读要点

Reconstructs stable hand-object grasping states from monocular egocentric videos for the first time

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2312.15719) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **22 Hand-Object Interaction & Fine-grained Activity** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Get a Grip」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **22 Hand-Object Interaction & Fine-grained Activity**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2312_15719_get-a-grip-reconstructing-hand-object-st.md`](../../sources/papers/sun_awesome_ego_2312_15719_get-a-grip-reconstructing-hand-object-st.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2312.15719>
- [arXiv PDF](https://arxiv.org/pdf/2312.15719)：已通读方法部分（实验与附录以略读为准）：§3 方法部分（稳定抓握定义、1-DoF 假设、重建优化；数据集标注与实验略读）

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2312.15719)
