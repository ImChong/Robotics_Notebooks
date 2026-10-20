---
type: entity
tags:
  - dataset
  - egocentric
  - collaboration
  - mixed-reality
status: draft
updated: 2026-10-20
summary: "头显采集的双人协作数据：350 对指导者-执行者，约 169 小时，含对话与动作标注。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../comparisons/humannet-table1-human-video-corpora.md
  - ../methods/vla.md
  - ../concepts/embodied-scaling-laws.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# HoloAssist

## 一句话定义

**HoloAssist**：执行者佩戴混合现实头显完成实物任务、指导者实时口头指导的第一视角交互数据集。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MR | Mixed Reality | 混合现实 |
| HMD | Head-Mounted Display | 头戴显示设备 |
| IMU | Inertial Measurement Unit | 惯性测量单元 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Microsoft + ETH Zurich 等 |
| 时间 | 2023-09（ICCV 2023） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 350 对指导者-执行者；约 169 小时 |
| 模态 | RGB、深度、头部位姿、手部三维姿态、眼动、音频、IMU |
| 基准 | 错误检测、介入类型预测、手部动作预测 |

## 为什么重要

- 记录人类如何纠错、何时介入，对人机协作与交互指导研究有直接参考价值。

## 核心原理

执行者视角与指导语音同步记录，附动作与对话标注。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | 项目页下载入口（文章给出链接） |

## 局限与风险

- 文章中同一数据集的模态数量表述不一致（列表写 7 路，图注写 8 类），以原论文为准。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [humannet-table1-human-video-corpora](../comparisons/humannet-table1-human-video-corpora.md)
- [vla](../methods/vla.md)
- [embodied-scaling-laws](../concepts/embodied-scaling-laws.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://holoassist.github.io/)
