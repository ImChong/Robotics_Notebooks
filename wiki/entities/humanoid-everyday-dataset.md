---
type: entity
title: Humanoid Everyday（开放世界人形操作数据集）
tags: [dataset, humanoid, manipulation, loco-manipulation, teleoperation, multimodal, usc, tri]
summary: "USC/TRI 大规模人形真机操作集：260 任务、10.3k 轨迹、300 万+ 帧多模态（RGB/深度/LiDAR/触觉+语言），含 loco-manipulation 与人–机交互，附云端标准化评测平台。"
updated: 2026-06-16
status: complete
related:
  - ../tasks/loco-manipulation.md
  - ../tasks/teleoperation.md
  - ../comparisons/humanoid-reference-motion-datasets.md
  - ../concepts/embodied-data-cleaning.md
sources:
  - ../../sources/sites/humanoideveryday.md
---

# Humanoid Everyday

**Humanoid Everyday**（Zhao et al., arXiv:[2510.08807](https://arxiv.org/abs/2510.08807)，2025）是 USC 与 Toyota Research Institute 发布的 **大规模人形机器人开放世界操作数据集**：通过高效 **人监督遥操** 采集真机多模态轨迹，覆盖从灵巧操作到 **下肢 loco-manipulation** 与 **人–机交互**，并提供 **云端评测平台** 供策略标准化部署与反馈。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TRI | Toyota Research Institute | 丰田研究院，本数据集合作方之一 |
| RGB-D | RGB + Depth | 彩色与深度多模态传感 |
| LiDAR | Light Detection and Ranging | 激光雷达，场景几何感知 |
| VLA | Vision-Language-Action | 视觉–语言–动作端到端策略范式 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Loco-Manipulation | Locomotion + Manipulation | 移动中同时完成操作的任务形态 |

## 为什么重要

- **真机人形数据而非人体 MoCap**：与 [AMASS](./amass.md)、[OMOMO](./omomo-dataset.md) 等 **人类参考运动** 不同，本集直接记录 **人形机器人执行** 的轨迹，省去重定向环节即可研究 manipulation / loco-manipulation 学习。
- **任务与传感广度**：**260 任务 / 7 大类**（含可变形物体、关节物体、工具使用、高精度操作等），**RGB + 深度 + LiDAR + 触觉 + 语言**，弥补既有集偏固定台架或缺下肢运动的问题。
- **评测基础设施**：除开源数据外提供 **cloud evaluation**，降低「各实验室环境不一致导致不可比」的摩擦。

## 核心信息

| 字段 | 内容 |
|------|------|
| 轨迹 | **10.3k** 条 |
| 帧数 | **3M+** @ **30 Hz** |
| 任务 | **260** 项 · **7** 大类 |
| 大类示例 | Loco-Manipulation、可变形/关节物体、工具使用、高精度操作、人–机交互等 |
| 项目页 | <https://humanoideveryday.github.io/> |
| 论文 | <https://arxiv.org/abs/2510.08807> |

## 流程总览（数据与评测）

```mermaid
flowchart LR
  tele[人监督遥操采集]
  data[多模态轨迹<br/>RGB / 深度 / LiDAR / 触觉 / 语言]
  train[策略学习<br/>论文内代表性方法分析]
  cloud[云端评测平台]
  tele --> data --> train
  train --> cloud
```

## 常见误区或局限

- **不是运动参考库**：不宜与 AMASS / PHUMA 等 **tracking 参考轨迹** 混用；定位是 **端到端操作/ loco-manipulation 学习数据**。
- **与 Paper Notebooks 深读分工**：姊妹仓库深读笔记尚未完成时，本页提供 **跨主题知识库级** 数据集归纳；细粒度实验表以论文与项目页为准。
- **泛化边界**：开放世界任务广，但采集平台与机器人形态仍决定 sim2real 与跨机型迁移上限。

## 与其他页面的关系

- **论文索引占位**：[Humanoid Everyday（Paper Notebooks）](./humanoid-everyday-dataset.md)
- **任务域**：[Loco-Manipulation](../tasks/loco-manipulation.md)、[Teleoperation](../tasks/teleoperation.md)
- **选型对照**：[humanoid-reference-motion-datasets](../comparisons/humanoid-reference-motion-datasets.md)

## 参考来源

- [Humanoid Everyday 项目页归档](../../sources/sites/humanoideveryday.md)
- 项目页：<https://humanoideveryday.github.io/>

## 关联页面

- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [Teleoperation](../tasks/teleoperation.md)
- [AMASS](./amass.md) — 人体 MoCap 参考库对照
- [PHUMA](./dataset-bfm-phuma.md) — 预重定向 G1 locomotion 对照

## 推荐继续阅读

- [Humanoid Everyday 项目页](https://humanoideveryday.github.io/) — 任务分布、技术摘要视频与云端评测入口
- [RoboCasa 大规模日常任务仿真](https://arxiv.org/abs/2406.02545) — 仿真侧日常操作基准对照
