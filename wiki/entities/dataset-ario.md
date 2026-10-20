---
type: entity
tags:
  - dataset
  - real-robot
  - multi-embodiment
  - data-standard
status: draft
updated: 2026-10-20
summary: "统一具身数据标准与大规模汇集数据集：约 300 万片段、258 个系列、覆盖 5 类模态。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/open-x-embodiment.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# ARIO（All Robots In One）

## 一句话定义

**ARIO**：既是统一具身数据组织标准，也是按该标准汇集的大规模多机器人数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ARIO | All Robots In One | 数据集名称 |
| PC | Point Cloud | 3D 点云 |
| TS | Timestamp | 统一时间戳 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 鹏城实验室等 |
| 时间 | 2024-08 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 约 300 万片段；258 个系列；321,064 项任务 |
| 模态 | 2D 图像、3D 点云、声音、文本、触觉 |
| 来源 | 真实采集、仿真平台、开源数据转换三类 |

## 为什么重要

- 提供 collection / series / task / episode 四级结构与统一时间戳，是跨数据集对齐的数据标准样例。

## 核心原理

各传感器按统一时间戳对齐，数据按四级结构组织。

## 工程实践

| 项 | 说明 |
|----|------|
| 下载 | 启智社区（openi.pcl.ac.cn，文章给出链接） |

## 局限与风险

- 三类来源质量差异较大，使用前需按来源筛选（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [open-x-embodiment](../concepts/open-x-embodiment.md)
- [manipulation](../tasks/manipulation.md)
- [imitation-learning](../methods/imitation-learning.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://ario-dataset.github.io/)
