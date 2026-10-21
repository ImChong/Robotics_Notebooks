---
type: entity
tags:
  - dataset
  - hand-object
  - egocentric
  - tracking
status: draft
updated: 2026-10-20
summary: "头戴设备采集的手-物三维追踪数据：833 分钟、370 万张以上图像、33 种刚体物体。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/manipulation.md
  - ../methods/diffusion-policy.md
  - ../methods/vla.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# HOT3D

## 一句话定义

**HOT3D**：面向第一视角三维手部与物体追踪的公开数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MANO | Hand Model with Articulated and Non-rigid Deformations | 参数化手部模型 |
| PBR | Physically Based Rendering | 基于物理的渲染材质 |
| Aria | Project Aria | Meta 研究眼镜 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Meta Reality Labs |
| 时间 | 2024-11（CVPR 2025） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 833 分钟；370 万张以上图像；19 名参与者；33 种刚体物体 |
| 设备 | Project Aria 眼镜与 Quest 3 头显 |
| 格式 | 手部提供 UmeTrack 与 MANO 格式；物体为带 PBR 材质扫描网格 |

## 为什么重要

- 真值由光学动作捕捉获得，并配有 Aria 与 Quest 3 双设备同步数据，是头戴式追踪的高质量基准。

## 核心原理

多视角 RGB、单色与眼动数据同步，三维位姿由光学动捕标定。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | projectaria_tools 开放数据页（文章给出链接） |

## 局限与风险

- 物体为 33 种刚体，形变与铰接物体不在覆盖范围（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [manipulation](../tasks/manipulation.md)
- [diffusion-policy](../methods/diffusion-policy.md)
- [vla](../methods/vla.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://facebookresearch.github.io/hot3d/)
