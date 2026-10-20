---
type: entity
tags:
  - dataset
  - sim
  - cross-embodiment
  - benchmark
status: draft
updated: 2026-10-20
summary: "统一格式仿真机器人数据集：约 50 万条轨迹、276 类任务，配套 MetaSim 多模拟器统一接口。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/embodied-scaling-laws.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# RoboVerse Data

## 一句话定义

**RoboVerse Data**：RoboVerse 平台的统一格式仿真数据，由仿真平台、合成数据集和统一基准三部分组成。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MetaSim | Meta Simulator（文中名） | 统一仿真接口 |
| IL | Imitation Learning | 模仿学习 |
| RL | Reinforcement Learning | 强化学习 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | UC Berkeley 等 |
| 时间 | 2025-04 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 约 50 万条高保真轨迹；276 类任务；约 5,500 个资产 |
| 状态转移 | 5,000 万条以上 |
| 接口 | MetaSim 统一多模拟器 |

## 为什么重要

- 通过 MetaSim 把多模拟器抽象为统一接口，使同一任务可在不同仿真与本体间迁移。

## 核心原理

把已有基准迁移到统一框架，数据集采用统一格式，并按泛化层级评测。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | Hugging Face RoboVerseOrg/roboverse_data |
| 代码 | RoboVerse 仓库 |

## 局限与风险

- 多模拟器统一会损失部分模拟器特有的物理细节（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [embodied-scaling-laws](../concepts/embodied-scaling-laws.md)
- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 评测基准在选型闭环中的定位与取舍

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://roboverseorg.github.io/)
