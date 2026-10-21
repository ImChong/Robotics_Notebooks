---
type: entity
tags:
  - dataset
  - real-robot
  - multi-embodiment
  - bimanual
status: draft
updated: 2026-10-20
summary: "多本体双臂数据集：超 18 万示范、15 种平台、421 项双臂任务，附层级能力金字塔标注。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/bimanual-manipulation.md
  - ../tasks/teleoperation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# RoboCOIN

## 一句话定义

**RoboCOIN**：覆盖 15 种双臂、半人形与人形本体的大规模多本体示范数据集，并配套数据处理流程。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RTML | Robot Trajectory Markup Language | 机器人轨迹标记语言（文章名称） |
| QC | Quality Control | 质检流程 |
| DataManager | RoboCOIN-DataManager | 配套仓库名 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 北京智源人工智能研究院等 |
| 时间 | 2025-11 |
| 模态 | 多模态示范（RGB、轨迹、文本标注；文章未列出具体传感器） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 超 18 万条示范；421 项双臂任务；39 种双臂协作动作；432 类物体 |
| 场景 | 住宅、商业、工业等 16 类环境 |
| 工具 | CoRobot 处理流程，基于机器人轨迹标记语言 RTML |

## 为什么重要

- 以“轨迹 / 片段 / 帧”三级标注组织能力层级，便于研究分层能力学习。

## 核心原理

CoRobot 流程对轨迹做质检、自动标注与多本体数据管理。

## 工程实践

| 项 | 说明 |
|----|------|
| 代码 | FlagOpen/RoboCOIN-DataManager（文章给出链接） |

## 局限与风险

- 多本体数据的分布不均衡程度需以官方统计为准（文章未展开）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [bimanual-manipulation](../tasks/bimanual-manipulation.md)
- [teleoperation](../tasks/teleoperation.md)
- [imitation-learning](../methods/imitation-learning.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://huggingface.co/RoboCOIN)
