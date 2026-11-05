---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2205.01791"
venue: "2022"
code: https://github.com/castacks/tartan_drive
summary: "RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2205_01791_tartandrive-a-large-scale-dataset-for-le.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# TartanDrive

**TartanDrive: A Large-Scale Dataset for Learning Off-Road Dynamics Models** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 125/564** 篇，分组 **Datasets**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

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
    P["§III-A ATV 平台: Yamaha Viking ATV<br/>人用 joystick 遥操作油门与转向, 刹车由人类驾驶员直接控制"] --> S["传感器 (ROS 车载计算机)<br/>Multisense S21 stereo camera (RGB/depth/IMU)<br/>NovAtel GNSS 全局位姿<br/>Racepak 数据记录仪 (shock travel / wheel rpm / 刹车位置)"]
    S --> D["§III-B 数据采集: rocks / mud / foliage 等多种地形<br/>人工遥操作, 驾驶员踩刹车干预即结束该 trajectory<br/>共 630 条 trajectories, 约 5 小时"]
    D --> M["派生模态<br/>stereo → 点云 (TartanVO 估计相机运动) → 俯视投影分箱<br/>→ top-down height map 与 RGB map (0.02 m/pixel, 500x500, 10 Hz)"]
    D --> Q["记录 intervention: 刹车位置 + 布尔干预信号"]
    M --> DS["TartanDrive 数据集 (Table II 各模态)"]
    Q --> DS
    DS --> V["§III-C Dynamical Variation<br/>10000 个长 10 的随机子序列 → 时间序列聚类 (欧氏距离)<br/>t-SNE 可视化 state displacement, 检验同一 action 序列是否产生不同轨迹"]
    DS --> B["§IV 多模态动力学建模 benchmark<br/>latent-space model: encoder (多模态 product of experts) → GRU latent model → decoder<br/>三种训练损失: State Reconstruction / Reconstruction / Contrastive"]
    B --> E["§V 实验: 损失类型是否有影响; 增加模态是否有帮助<br/>(对比 kinematic bicycle model 基线; 原始与更困难评测集)"]
```

图依据：数据采集流程图，论文 §III The Dataset（§III-A–C、Fig. 3、Fig. 5、Table I–II）、§IV Multi-modal Modeling、§V Experiments and Analysis。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 125/564 |
| 分组 | Datasets |
| 出处 | 2022 |
| 论文 | <https://arxiv.org/abs/2205.01791> |
| 代码/项目 | <https://github.com/castacks/tartan_drive> |
| 子类 / 象限 | 导航与驾驶示范数据 · 多传感器与空间标注 · 运动预测与规划数据 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（Datasets）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2205.01791) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Datasets** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「TartanDrive」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/rcl_awesome_wam_2205_01791_tartandrive-a-large-scale-dataset-for-le.md`](../../sources/papers/rcl_awesome_wam_2205_01791_tartandrive-a-large-scale-dataset-for-le.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2205.01791>
- [arXiv PDF](https://arxiv.org/pdf/2205.01791)：已通读 §III The Dataset、§IV Multi-modal Modeling、§V Experiments and Analysis、§VI Conclusions；§I–II 与附录以略读为准

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2205.01791)
