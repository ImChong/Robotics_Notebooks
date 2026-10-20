---
type: entity
tags:
  - dataset
  - tactile
  - visuo-tactile
  - in-the-wild
status: draft
updated: 2026-10-20
summary: "自然场景视觉触觉数据：采集者手持 GelSight 传感器按压各种物体，同步录制第一视角视频。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# Touch and Go

## 一句话定义

**Touch and Go**：在野外场景采集的成对视觉与触觉数据集，附材质类别与触碰起始时刻标注。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GelSight | Gel-based tactile sensor | 基于凝胶的视觉触觉传感器 |
| VT | Visuo-Tactile（视觉触觉） | 视觉与触觉联合 |
| NeurIPS | Neural Information Processing Systems | 会议 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | University of Michigan、CMU 等 |
| 时间 | 2022-11（NeurIPS 2022 数据集赛道） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 采集 | 手持 GelSight 触觉传感器 + 第一视角视频 |
| 标注 | 材质类别、触碰起始时刻 |
| 任务 | 自监督视觉触觉特征学习、触觉驱动图像风格化、未来触觉预测 |

## 为什么重要

- 突破实验室与仿真限制，覆盖大量“野外”物体与场景。

## 核心原理

人类手持传感器自然按压，形成视觉与触觉的时间对齐样本。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | touch-and-go.github.io 下载页（文章给出链接） |
| 代码 | fredfyyang/Touch-and-Go 仓库 |

## 局限与风险

- 手持采集的力与压强控制不统一，定量比较需谨慎（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [visuo-tactile-fusion](../concepts/visuo-tactile-fusion.md)
- [manipulation](../tasks/manipulation.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://touch-and-go.github.io/)
