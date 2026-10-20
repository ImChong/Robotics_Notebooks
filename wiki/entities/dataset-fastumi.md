---
type: entity
tags:
  - dataset
  - real-robot
  - human-demo
status: draft
updated: 2026-10-20
summary: "便携 UMI 式手持示范采集接口配套数据：首版 1 万余条、扩展版 10 万余条真实示范。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/open-x-embodiment.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# FastUMI / FastUMI-100K

## 一句话定义

**FastUMI**：可复用的手持式人类示范采集接口及其配套真实示范数据集（FastUMI-Data / FastUMI-100K）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| UMI | Universal Manipulation Interface | 手持夹爪式示范采集接口 |
| EE | End-Effector | 末端执行器位姿 |
| VIO | Visual-Inertial Odometry | 视觉惯性里程计（FastUMI 改用现成跟踪模块替代） |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 上海人工智能实验室等 |
| 时间 | 2024-09（100K 版 2025-10） |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | FastUMI-Data 1 万余条（22 项任务）；FastUMI-100K 10 万余条（54 项任务） |
| 模态 | 末端状态、多视角腕部鱼眼图像、文本标注 |
| 获取 | Hugging Face 数据页（文章给出链接） |

## 为什么重要

- 以低部署门槛的手持采集扩大真实示范规模，是「人类示范代替遥操作」路线的代表数据。

## 核心原理

每条轨迹约 120–500 帧，同步记录末端状态与腕部鱼眼视频，并附文本标注，可直接用于模仿学习。

## 工程实践

| 项 | 说明 |
|----|------|
| 许可 | 文章未注明，使用前以官方页为准 |
| 用途 | 腕部视角模仿学习；长程与双臂任务评测 |

## 局限与风险

- 数据来自人类手持采集，与具体机器人本体的动作空间需额外对齐（推测，待官方文档核实）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [open-x-embodiment](../concepts/open-x-embodiment.md)
- [manipulation](../tasks/manipulation.md)
- [imitation-learning](../methods/imitation-learning.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://huggingface.co/datasets/fastumi100k/fastumi100k)
