---
type: entity
tags:
  - dataset
  - 3d-scene
  - embodied-sim
  - navigation
status: draft
updated: 2026-10-20
summary: "基于真实空间三维扫描的具身仿真数据库：572 栋建筑、1,447 个楼层、21.1 万平方米。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../entities/habitat-sim.md
  - ../tasks/vision-language-navigation.md
  - ../entities/matterport3d-simulator.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# Gibson Environment Dataset

## 一句话定义

**Gibson**：以真实室内空间重建为基础的具身智能仿真环境与数据库。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Goggles | 文中合成机制名 | 缩小仿真与真实视觉差距 |
| Sim2Real | Simulation to Real | 仿真到真实迁移 |
| RGB-D | RGB + Depth | 彩色与深度 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Stanford University、UC Berkeley |
| 时间 | 2018-08（CVPR 2018） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 572 栋建筑；1,447 个楼层；21.1 万平方米 |
| 模态 | 三维重建、RGB、深度、表面法线，部分有语义标注 |
| 演进 | 后续发展为 iGibson 与 BEHAVIOR 等平台 |

## 为什么重要

- 通过 Goggles 合成机制缩小仿真与真实相机画面的差距，是 Sim2Real 视觉导航的早期代表。

## 核心原理

真实空间扫描重建后放入仿真，智能体受物理与空间约束运行。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | gibsonenv.stanford.edu 数据库（文章给出链接） |
| 代码 | StanfordVL/GibsonEnv |

## 局限与风险

- 主要面向视觉导航，操作与接触类任务支持有限（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [habitat-sim](../entities/habitat-sim.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [matterport3d-simulator](../entities/matterport3d-simulator.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](http://gibsonenv.stanford.edu/database/)
