---
type: entity
tags:
  - dataset
  - 3d-scene
  - rgb-d
  - semantic-segmentation
status: draft
updated: 2026-10-20
summary: "大规模室内 RGB-D 扫描数据集：1,513 次扫描、250 万帧视图，含位姿、重建与实例语义标注。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../entities/habitat-sim.md
  - ../tasks/vision-language-navigation.md
  - ../entities/matterport3d-simulator.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# ScanNet

## 一句话定义

**ScanNet**：由自研 RGB-D 采集系统重建、众包标注的室内三维场景数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RGB-D | RGB + Depth | 彩色与深度 |
| CAD | Computer-Aided Design | CAD 模型检索任务 |
| 3D | Three-Dimensional | 三维 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Stanford University 等 |
| 时间 | 2017-02（CVPR 2017） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 1,513 次扫描；250 万帧视图 |
| 标注 | 三维相机位姿、表面重建、实例级语义分割 |
| 获取 | 需签署使用协议后下载 |

## 为什么重要

- 三维场景理解的通用基准，ScanRefer、EmbodiedScan 等数据均以其场景为基础。

## 核心原理

RGB-D 视频自动重建，再经众包完成实例语义标注。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | ScanNet 官方签署协议（文章给出链接） |

## 局限与风险

- 需签协议且扫描场景以住宅为主（推测），与机器人真实部署场景存在差异。
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

- [官方数据页](http://www.scan-net.org/)
