---
type: entity
tags:
  - dataset
  - motion-capture
  - human-motion
  - social
status: draft
updated: 2026-10-20
summary: "多相机舞台采集的三维人体动作数据：439 名参与者、500 小时、追踪超 5,400 万帧。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../comparisons/humannet-table1-human-video-corpora.md
  - ../methods/vla.md
  - ../concepts/embodied-scaling-laws.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# Embody 3D

## 一句话定义

**Embody 3D**：包含单人动作与多人社交互动的大规模三维动作数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MoCap | Motion Capture | 动作捕捉 |
| 3D | Three-Dimensional | 三维 |
| SOC | Social interaction（缩写为本文约定） | 多人社交交互 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Meta Codec Avatars Lab |
| 时间 | 2025-10 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 439 名参与者；500 小时；追踪帧数超 5,400 万 |
| 模态 | 人体运动（含手部与体型）、文本标注、每人独立音轨 |
| 划分 | 动作、手部交互、行走、双人对话、多人对话、情景任务、一日生活共 7 部分 |

## 为什么重要

- 覆盖多人对话与共同生活场景，可用于研究社交行为与全身动作生成。

## 核心原理

多相机采集舞台重建三维人体网格并追踪，附文本与音轨标注。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | 填写申请表后下载（文章给出链接） |

## 局限与风险

- 数据偏向人体动作与社交，与机器人直接操作的衔接需额外重定向（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [humannet-table1-human-video-corpora](../comparisons/humannet-table1-human-video-corpora.md)
- [vla](../methods/vla.md)
- [embodied-scaling-laws](../concepts/embodied-scaling-laws.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://github.com/facebookresearch/embody-3d)
