---
type: entity
tags:
  - dataset
  - tactile
  - rgb-d
  - reconstruction
status: draft
updated: 2026-10-20
summary: "GelSight 与 Azure Kinect 在 YCB 物体上的仿真与真实视觉触觉数据，用于增量三维重建。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# YCB-Sight

## 一句话定义

**YCB-Sight**：包含 GelSight 触觉与 RGB-D 相机的 YCB 物体视觉触觉数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GP | Gaussian Process | 高斯过程 |
| RGB-D | RGB + Depth | 彩色与深度 |
| GelSight | Gel-based tactile sensor | 视觉触觉传感器 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Carnegie Mellon University（RoboTouch 实验室） |
| 时间 | 2021-09（随 ShapeMap 3-D，ICRA 2022） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 传感器 | GelSight 触觉传感器与 Azure Kinect RGB-D 相机 |
| 数据 | 仿真与真实交互各一套 |
| 方法 | 高斯过程（GP）空间因子图 |

## 为什么重要

- 用于 ShapeMap 3-D 的触觉局部形状恢复与高斯过程因子图建图，解决视觉被遮挡的问题。

## 核心原理

在仿真中训练模型从触觉图像恢复局部形状，再在因子图上增量推断隐式表面。

## 工程实践

| 项 | 说明 |
|----|------|
| 代码 | Robo-Touch/YCB-Sight 仓库（文章给出链接） |

## 局限与风险

- 数据规模与物体数量未在文章中给出（推测需看原仓库）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [visuo-tactile-fusion](../concepts/visuo-tactile-fusion.md)
- [manipulation](../tasks/manipulation.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://github.com/Robo-Touch/YCB-Sight)
