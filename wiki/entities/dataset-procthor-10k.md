---
type: entity
tags:
  - dataset
  - procedural
  - synthetic-scene
  - navigation
status: draft
updated: 2026-10-20
summary: "程序化生成的 1 万个可交互家庭环境，用于训练零样本的具身导航策略。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../entities/habitat-sim.md
  - ../tasks/vision-language-navigation.md
  - ../entities/matterport3d-simulator.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# ProcTHOR-10K

## 一句话定义

**ProcTHOR-10K**：由 ProcTHOR 框架按需采样生成的 1 万个可交互虚拟房屋。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ProcTHOR | Procedural THOR | 程序化生成框架 |
| AI2-THOR | AI2 Interactive Thor | 可交互仿真环境 |
| RGB | Red-Green-Blue | 仅用 RGB 输入的策略 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Allen Institute for AI |
| 时间 | 2022-06（NeurIPS 2022 杰出论文奖） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 1 万个生成房屋 |
| 结果 | 6 项具身基准上取得领先（文章） |
| 配套 | ProcTHOR 框架与 AI2-THOR 环境 |

## 为什么重要

- 证明大规模程序化场景足以支撑在多个具身基准上的强零样本表现。

## 核心原理

程序化采样户型与家具布局，生成大规模训练场景。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | procthor.allenai.org 数据页（文章给出链接） |

## 局限与风险

- “零样本超过使用下游训练数据的方法”为论文自述，需看具体基准设定（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [habitat-sim](../entities/habitat-sim.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [matterport3d-simulator](../entities/matterport3d-simulator.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://procthor.allenai.org/)
