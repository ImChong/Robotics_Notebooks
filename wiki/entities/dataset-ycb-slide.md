---
type: entity
tags:
  - dataset
  - tactile
  - sliding
  - localization
status: draft
updated: 2026-10-20
summary: "DIGIT 传感器在 YCB 物体上的真实与仿真滑动数据，支持触觉定位与建图。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# YCB-Slide

## 一句话定义

**YCB-Slide**：视觉触觉传感器在标准 YCB 物体表面滑动交互的数据集，附真值网格与高度图。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DIGIT | DIGIT tactile sensor | 视觉触觉传感器 |
| YCB | Yale-CMU-Berkeley Object Set | 标准物体集 |
| PF | Particle Filter | 粒子滤波 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Carnegie Mellon University、Meta AI |
| 时间 | 2022-10（随 MidasTouch，CoRL 2022） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 内容 | DIGIT 触觉图像、传感器位姿、RGB 视频、真值网格 |
| 仿真部分 | 真值高度图与接触掩码 |
| 方法 | 蒙特卡洛粒子滤波（PF）与触觉编码网络 |

## 为什么重要

- 配合 MidasTouch 在无视觉先验下做在线全局定位，是触觉定位的代表数据。

## 核心原理

持续估计传感器在物体表面的位姿分布，实现在线全局定位。

## 工程实践

| 项 | 说明 |
|----|------|
| 代码 | rpl-cmu/YCB-Slide 仓库（文章给出链接） |

## 局限与风险

- 依赖 DIGIT 传感器与特定物体集，通用性需验证（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [visuo-tactile-fusion](../concepts/visuo-tactile-fusion.md)
- [manipulation](../tasks/manipulation.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://github.com/rpl-cmu/YCB-Slide)
