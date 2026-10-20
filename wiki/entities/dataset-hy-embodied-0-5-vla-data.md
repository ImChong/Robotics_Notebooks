---
type: entity
tags:
  - dataset
  - real-robot
  - bimanual
  - vla
status: draft
updated: 2026-10-20
summary: "腾讯 HyVLA-0.5 预训练用双臂操作数据：指尖式 UMI 采集加光学动捕，本次开源 2,163 小时。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/open-x-embodiment.md
  - ../tasks/manipulation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# Hy-Embodied-0.5-VLA-Data

## 一句话定义

**Hy-Embodied-0.5-VLA-Data**：面向 VLA 预训练的双臂操作数据集，随 HyVLA-0.5 技术报告发布。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作模型 |
| UMI | Universal Manipulation Interface | 手持采集接口 |
| LeRobot | LeRobot 机器人学习库 | 数据格式与工具链 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 腾讯 Robotics X / 混元 |
| 时间 | 2026-06 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 完整语料 1 万小时以上；本次开源约 20%（2,163 小时、25 万余片段） |
| 任务 | 70 余项操作任务 |
| 格式 | LeRobot v3.0 结构 |

## 为什么重要

- 展示 UMI 式采集在 VLA 预训练规模上的可行性，且采用 LeRobot v3.0 组织，便于复用。

## 核心原理

自研指尖式 UMI 设备采集，光学动捕记录轨迹；完整语料预训练 HyVLA-0.5。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源比例 | 约 20%（文章数字） |
| 迁移 | 文章称可迁移到 4 种真实平台 |

## 局限与风险

- 公开子集仅为完整语料的约 20%，复现完整结果受限（文章信息）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [open-x-embodiment](../concepts/open-x-embodiment.md)
- [manipulation](../tasks/manipulation.md)
- [imitation-learning](../methods/imitation-learning.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://huggingface.co/datasets/tencent/Hy-Embodied-0.5-VLA-Data)
