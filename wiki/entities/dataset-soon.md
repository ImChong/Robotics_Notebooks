---
type: entity
tags:
  - dataset
  - navigation
  - object-goal
  - vln
status: draft
updated: 2026-10-20
summary: "场景描述驱动的目标导航数据（FAO）：3,848 组指令，允许从任意起点找到目标。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/vision-language-navigation.md
  - ../entities/habitat-sim.md
  - ../entities/matterport3d-simulator.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# SOON（FAO 数据集）

## 一句话定义

**SOON**：只依据目标物体及周边环境描述、从任意起点寻找目标的导航任务，配套 FAO 数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FAO | From Anywhere to Object | 数据集名称，任意起点到物体 |
| GBE | Graph-Based Exploration（文中方法名） | 图探索方法 |
| R2R | Room-to-Room | 对照的路线指令导航数据 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 莫纳什大学、中山大学等 |
| 时间 | 2021-03（CVPR 2021） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 3,848 组指令；词表 1,649；平均每组 38.6 词 |
| 划分 | 训练集 3,085 组指令、28,015 条轨迹，覆盖 38 栋房屋 |
| 描述维度 | 物体属性、物体关系、所在区域、相邻区域 |

## 为什么重要

- 不依赖逐步路线指令，更接近真实的“找东西”需求。

## 核心原理

指令描述目标与环境关系，智能体在场景图上探索并定位目标。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | GitHub 仓库（文章给出链接） |

## 局限与风险

- 指令为描述性而非精确路径，评测结果对文本质量敏感（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [habitat-sim](../entities/habitat-sim.md)
- [matterport3d-simulator](../entities/matterport3d-simulator.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://github.com/ZhuFengdaaa/SOON)
