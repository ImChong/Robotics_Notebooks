---
type: entity
tags:
  - dataset
  - 3d-scene
  - navigation
  - habitat
status: draft
updated: 2026-10-20
summary: "1,000 个建筑级三维重建场景，可导航面积 11.25 万平方米，是 Habitat 导航训练的主流场景库。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../entities/habitat-sim.md
  - ../tasks/vision-language-navigation.md
  - ../entities/matterport3d-simulator.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# HM3D（Habitat-Matterport 3D）

## 一句话定义

**HM3D**：由真实建筑扫描重建、可直接在 Habitat 模拟器中使用的大规模三维室内场景集合。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HM3D | Habitat-Matterport 3D | 数据集名称 |
| Habitat | Habitat 仿真平台 | Meta 的具身仿真平台 |
| RGB-D | RGB + Depth | 彩色与深度图像 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Meta AI、Matterport 等 |
| 时间 | 2021-09（NeurIPS 2021） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 1,000 个建筑级场景；可导航面积 11.25 万平方米 |
| 对比 | 面积为 Matterport3D、Gibson 的 1.4 至 3.7 倍（文章） |
| 后续 | HM3D-Semantics 语义版 |

## 为什么重要

- 场景数量与可导航面积显著大于 Matterport3D 与 Gibson，是导航智能体训练的常用基础资产。

## 核心原理

对真实建筑做三维重建，渲染逼真度高于同类扫描数据（文章称 20%–85%）。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | aihabitat.org 数据页（文章给出链接） |
| 仿真 | habitat-sim 仓库 |

## 局限与风险

- 渲染逼真度的量化数字来自论文自述，未经独立核对。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [habitat-sim](../entities/habitat-sim.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [matterport3d-simulator](../entities/matterport3d-simulator.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://aihabitat.org/datasets/hm3d/)
