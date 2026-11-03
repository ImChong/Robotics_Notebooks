---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2405.14731"
venue: "IEEE Robotics and Automation Letters"
summary: "RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2405_14731_coped-advancing-multi-robot-collaborativ.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# CoPeD-Advancing Multi-Robot Collaborative Perception

**CoPeD-Advancing Multi-Robot Collaborative Perception: A Comprehensive Dataset in Real-World Environments** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 102/564** 篇，分组 **Datasets**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  A["采集平台：3 台地面机器人 + 2 台空中机器人<br/>ROS 1，5G Wi-Fi 子网，NTP 时间同步"] --> B["场景序列<br/>室内 Indoor-NYUARPL / 室外 HOUSE、FOREST<br/>机器人队拆成「1 空 + 1 地」子队"]
  B --> C["原始多模态数据流<br/>相机 / LiDAR / IMU / GPS，多速率"]
  C --> D["标定"]
  D --> D1["空中机器人：Kalibr<br/>相机与 IMU 内外参"]
  D --> D2["地面机器人：graph-based 多传感器标定<br/>Ouster LiDAR 为根坐标系"]
  D --> D3["空地标定：<br/>标定板对齐前向相机外参"]
  D1 --> E["位姿估计"]
  D2 --> E
  D3 --> E
  E --> E1["空中：GPS + OpenVINS 双目 VIO"]
  E --> E2["地面：IMU + 轮式里程计"]
  E --> E3["跨机器人相对位姿：<br/>AprilTag bundle + PnP"]
  C --> F["可选高层感知标注（零样本）"]
  F --> F1["语义：RAM + Grounding DINO + SAM<br/>得 2D 实例掩码，再后处理出 2D 框"]
  F1 --> F2["时序传播：关键帧分割传播到后续帧"]
  F --> F3["深度：ZoeDepth 单目零样本深度"]
  E1 --> G["数据集内容：原始传感器流 + 位姿 + 可选标注"]
  E2 --> G
  E3 --> G
  F2 --> G
  F3 --> G
  G --> H["用途演示：GNN 特征图通信的<br/>协同深度估计 / 语义分割（传感器被破坏的场景）"]
```

图依据：数据采集与标注流程图，依据论文 §III（平台与序列）、§IV（标定）、§V（位姿估计与零样本标注）、§VII（用途演示）；箭头顺序按论文章节顺序与 §I 对贡献的叙述归纳，论文未单独给出总流程图。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 102/564 |
| 分组 | Datasets |
| 出处 | IEEE Robotics and Automation Letters |
| 论文 | <https://ieeexplore.ieee.org/document/10540310/> |
| 子类 / 象限 | 多机器人协作感知数据 · 多传感器与空间标注 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://ieeexplore.ieee.org/document/10540310/) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Datasets** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「CoPeD-Advancing Multi-Robot Collaborative Perception」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Datasets**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/rcl_awesome_wam_2405_14731_coped-advancing-multi-robot-collaborativ.md`](../../sources/papers/rcl_awesome_wam_2405_14731_coped-advancing-multi-robot-collaborativ.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://ieeexplore.ieee.org/document/10540310/>
- [arXiv PDF](https://arxiv.org/pdf/2405.14731)：已读与流程图相关的章节（其余部分以略读为准）：§I、§III（数据集描述）、§IV（标定）、§V（标注）、§VI（数据集属性）、§VII（用例）、§VIII；Tab. I–IV 与 Fig. 2–8 略读

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://ieeexplore.ieee.org/document/10540310/)
