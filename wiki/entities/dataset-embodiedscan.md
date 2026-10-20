---
type: entity
tags:
  - dataset
  - 3d-perception
  - rgb-d
  - embodied
status: draft
updated: 2026-10-20
summary: "第一视角多模态三维感知数据：5,000 余次扫描、100 万帧 RGB-D，标注 16 万个三维有向包围框。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../entities/habitat-sim.md
  - ../tasks/vision-language-navigation.md
  - ../entities/matterport3d-simulator.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# EmbodiedScan

## 一句话定义

**EmbodiedScan**：面向具身智能的第一视角多模态三维感知数据集与基准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RGB-D | RGB + Depth | 彩色与深度 |
| 3D | Three-Dimensional | 三维 |
| OCC | Occupancy（语义占用） | 稠密语义占用标注 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 上海人工智能实验室等 |
| 时间 | 2023-12（CVPR 2024） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 5,000 余次扫描；100 万帧 RGB-D；100 万条语言提示 |
| 标注 | 16 万个三维有向包围框；760 余类；80 类稠密占用 |
| 基线 | Embodied Perceptron |

## 为什么重要

- 同时提供三维检测框、语言提示与稠密语义占用，覆盖基础感知与语言引导感知两类基准。

## 核心原理

第一视角 RGB-D 序列重建三维场景，并附语言描述与三维框标注。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | GitHub 仓库数据章节（文章给出链接） |

## 局限与风险

- 数据主要服务感知基准，直接用于机器人控制仍需额外策略数据（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [habitat-sim](../entities/habitat-sim.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [matterport3d-simulator](../entities/matterport3d-simulator.md)
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) — 本页感知能力在机器人视觉感知栈中的位置与选型

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://tai-wang.github.io/embodiedscan/)
