---
type: entity
tags:
  - dataset
  - real-robot
  - human-robot-interaction
status: draft
updated: 2026-10-20
summary: "人机共处操作数据：协作者、同事、监督者三类角色各 20 项任务，共 160 余小时。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/loco-manipulation.md
  - ../tasks/teleoperation.md
  - ../methods/vla.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# HABIT

## 一句话定义

**HABIT**：面向人机共处环境的机器人操作数据集，按人与机器人的三种关系组织任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HRI | Human-Robot Interaction | 人机交互 |
| KAIST | Korea Advanced Institute of Science and Technology | 韩国科学技术院 |
| NeurIPS | Neural Information Processing Systems | 会议 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Config + KAIST |
| 时间 | 2026-06 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 1 万余条片段；160 余小时；60 项任务（每类角色 20 项） |
| 模态 | 人机同场的真机操作片段（视觉观测 + 机器人动作）；具体传感器配置文章未注明，以官方页为准 |
| 角色 | 协作者、同事、监督者 |
| 评测 | 文章称入选 NeurIPS 2026 评测与数据集赛道口头报告 |

## 为什么重要

- 包含仅靠无人数据学不到的人类感知行为（如时空同步、让行、手势理解），聚焦人机交互。

## 核心原理

按三类人机关系分别采集，并用于比较策略学习到的人类感知行为。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | Hugging Face 数据页（文章给出链接） |

## 局限与风险

- 任务规模较小（60 项），更适合研究人机交互行为而非通用预训练（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [loco-manipulation](../tasks/loco-manipulation.md)
- [teleoperation](../tasks/teleoperation.md)
- [vla](../methods/vla.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://habit-dataset.github.io/)
