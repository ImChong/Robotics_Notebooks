---
type: entity
tags:
  - dataset
  - tactile
  - cross-modal
  - gelsight
status: draft
updated: 2026-10-20
summary: "GelSight 与视觉的跨模态配对数据：300 万组视觉-触觉图像，支持两个方向的预测。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# VisGel

## 一句话定义

**VisGel**：机械臂装配 GelSight 与网络摄像头采集的视觉-触觉跨模态数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GelSight | Gel-based tactile sensor | 视觉触觉传感器 |
| CVPR | Conference on Computer Vision and Pattern Recognition | 会议 |
| MIT CSAIL | MIT Computer Science and Artificial Intelligence Laboratory | 麻省理工学院计算机科学与人工智能实验室 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | MIT CSAIL |
| 时间 | 2019-06（CVPR 2019） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 195 类物体；1.2 万次触碰；每次 250 帧；300 万组配对图像 |
| 任务 | 视觉到触觉、触觉到视觉预测 |
| 获取 | GitHub 仓库数据章节（文章给出链接） |

## 为什么重要

- 研究视觉与触觉之间的跨模态联系，提出从视觉合成触觉、从触觉想象画面两项任务。

## 核心原理

机械臂重复触碰物体，同时录制触觉与外部视频，并融合尺度与位置信息做条件预测。

## 工程实践

| 项 | 说明 |
|----|------|
| 代码 | YunzhuLi/VisGel 仓库 |

## 局限与风险

- 数据发布于 2019 年，传感器与物体类别偏旧（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [visuo-tactile-fusion](../concepts/visuo-tactile-fusion.md)
- [manipulation](../tasks/manipulation.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://github.com/YunzhuLi/VisGel)
