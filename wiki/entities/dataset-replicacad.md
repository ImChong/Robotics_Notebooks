---
type: entity
tags:
  - dataset
  - 3d-scene
  - interactive-assets
  - rearrangement
status: draft
updated: 2026-10-20
summary: "可交互公寓三维资产：Replica 扫描公寓的美术重建版本，含铰接家具与 105 种布局。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../entities/habitat-sim.md
  - ../tasks/vision-language-navigation.md
  - ../entities/matterport3d-simulator.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# ReplicaCAD

## 一句话定义

**ReplicaCAD**：对 Replica 中 FRL 公寓的美术重建，可在 Habitat 中进行物理交互的资产集合。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FRL | FRL 公寓（Replica 扫描空间名） | 数据来源空间 |
| HAB | 文中的家庭助手基准名 | 收拾、整理、摆餐桌等任务 |
| Habitat | Habitat 仿真平台 | Meta 具身仿真平台 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Meta AI |
| 时间 | 2021-06（随 Habitat 2.0） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 资产 | 90 余个带质量、摩擦与回弹参数的物体；可开合铰接家具 |
| 布局 | 105 种布局，其中 84 种公开 |
| 配套 | Habitat 2.0 与 HAB 家庭助手基准 |

## 为什么重要

- 为重排（rearrangement）与移动操作任务提供带物理参数的可交互家具与物体。

## 核心原理

美术重建后加入物理属性与铰接关节，配合布局变化生成任务多样性。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | aihabitat.org 数据页（文章给出链接） |

## 局限与风险

- 单一公寓空间，场景多样性有限（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [habitat-sim](../entities/habitat-sim.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [matterport3d-simulator](../entities/matterport3d-simulator.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://aihabitat.org/datasets/replica_cad/)
