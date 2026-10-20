---
type: entity
tags:
  - dataset
  - egocentric
  - exocentric
  - human-video
status: draft
updated: 2026-10-20
summary: "第一与第三视角同步的技能活动视频：1,286 小时、740 名参与者、13 座城市。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../comparisons/humannet-table1-human-video-corpora.md
  - ../methods/vla.md
  - ../concepts/embodied-scaling-laws.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# Ego-Exo4D

## 一句话定义

**Ego-Exo4D**：同步采集第一视角与多台第三视角视频的技能型活动数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric | 第一人称视角 |
| Exo | Exocentric | 第三人称视角 |
| IMU | Inertial Measurement Unit | 惯性测量单元 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Meta + 多所高校 |
| 时间 | 2023-11（CVPR 2024） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 1,286 小时视频；740 名参与者；123 种自然场景 |
| 模态 | 多声道音频、眼动、三维点云、相机位姿、IMU |
| 标注 | 教练点评、参与者讲解、原子动作描述三类语言标注 |

## 为什么重要

- 同时提供第一视角与外部视角，适合研究人类技能在两种视角下的对应关系。

## 核心原理

同一活动同步多机位与佩戴式设备采集，并配套多项理解与估计基准。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | 需按官网流程申请（文章给出链接） |

## 局限与风险

- 以视频理解为主，对机器人动作的直接可用性较弱（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [humannet-table1-human-video-corpora](../comparisons/humannet-table1-human-video-corpora.md)
- [vla](../methods/vla.md)
- [embodied-scaling-laws](../concepts/embodied-scaling-laws.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://ego-exo4d-data.org/)
