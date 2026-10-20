---
type: entity
tags:
  - dataset
  - navigation
  - dialog
  - vln
status: draft
updated: 2026-10-20
summary: "家庭场景中人与人对话导航数据：2,050 段对话，7,000 余条穿插问答的导航轨迹。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/vision-language-navigation.md
  - ../entities/habitat-sim.md
  - ../entities/matterport3d-simulator.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# CVDN（Cooperative Vision-and-Dialog Navigation）

## 一句话定义

**CVDN**：导航者向知道最短路径的“先知”提问、由其回答的人类对话导航数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLN | Vision-and-Language Navigation | 视觉语言导航 |
| CVDN | Cooperative Vision-and-Dialog Navigation | 本数据集名称 |
| CoRL | Conference on Robot Learning | 会议 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 华盛顿大学 |
| 时间 | 2019-07（CoRL 2019） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 2,050 段对话；7,000 余条导航轨迹；83 个房屋扫描 |
| 环境 | 逼真家庭仿真环境 |
| 任务 | 基于对话历史的导航 |

## 为什么重要

- 研究如何利用更早的对话历史辅助导航决策，是对话式视觉导航的基础数据。

## 核心原理

人与人在线交互采集对话与轨迹，任务要求智能体结合目标与历史对话推断动作。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | cvdn.dev 下载页（文章给出链接） |

## 局限与风险

- 房屋数量有限（83 个），泛化结论需谨慎（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [habitat-sim](../entities/habitat-sim.md)
- [matterport3d-simulator](../entities/matterport3d-simulator.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://cvdn.dev/)
