---
type: entity
tags:
  - dataset
  - hand-object
  - egocentric
  - bimanual
status: draft
updated: 2026-10-20
summary: "双手与物体交互的多视角 RGB-D 数据，附双手三维姿态、物体 6D 位姿与交互标签。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/manipulation.md
  - ../methods/diffusion-policy.md
  - ../methods/vla.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# H2O Dataset（2 Hands and Objects）

## 一句话定义

**H2O**：首个借助双手姿态研究第一视角动作的基准数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| H2O | 2 Hands and Objects | 数据集名称 |
| 6D | Six-Degree-of-Freedom | 六自由度位姿 |
| RGB-D | RGB + Depth | 彩色与深度 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | ETH Zurich、Microsoft |
| 时间 | 2021-04（ICCV 2021） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 模态 | 同步多视角 RGB-D、交互标签、物体类别、双手三维姿态、物体 6D 位姿、相机位姿、物体网格、场景点云 |
| 标注 | 无标记三维标注方法逐帧生成 |
| 挑战 | 手部姿态估计、动作识别 |

## 为什么重要

- 同时提供双手与物体的三维真值，并设立手姿态估计与动作识别两项挑战。

## 核心原理

无标记三维标注为每帧生成双手与物体位姿，用于预测交互类别。

## 工程实践

| 项 | 说明 |
|----|------|
| 代码 | taeinkwon/h2odataset 仓库（文章给出链接） |

## 局限与风险

- 规模与物体种类未在文章中给出具体数字，需看原论文（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [manipulation](../tasks/manipulation.md)
- [diffusion-policy](../methods/diffusion-policy.md)
- [vla](../methods/vla.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://taeinkwon.com/projects/h2o/)
