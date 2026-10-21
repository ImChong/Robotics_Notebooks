---
type: entity
tags:
  - dataset
  - real-robot
  - bimanual
  - behavior-cloning
status: draft
updated: 2026-10-20
summary: "ABC 开源行为克隆体系的核心数据：约 3,500 小时、13 万余条双臂遥操作轨迹、195 项任务。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../tasks/bimanual-manipulation.md
  - ../tasks/teleoperation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# ABC-130K

## 一句话定义

**ABC-130K**：迄今声称最大的开源双臂遥操作数据集之一，同时附带仿真与真机评测数据。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BC | Behavior Cloning | 行为克隆 |
| VLA | Vision-Language-Action | 视觉-语言-动作模型 |
| ABC | 文中体系名缩写 | 全称文章未给出 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | UC Berkeley 等（含 Amazon FAR、MIT、XDOF） |
| 时间 | 2026-06 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 约 3,500 小时；13 万余条轨迹；195 项任务 |
| 模态 | 双臂遥操作轨迹（视觉观测 + 动作），含真机与仿真两部分；相机/传感器配置文章未注明，以官方页为准 |
| 仿真 | 20 个场景、400 小时仿真遥操作数据 |
| 评测 | 100 余小时真机评测数据与评分规则 |

## 为什么重要

- 配套低成本采集硬件、训练框架与仿真流程，覆盖从数据到评测的完整行为克隆链路。

## 核心原理

真实遥操作数据与仿真数据一并提供，并用评测数据比较扩散 Transformer 与 VLA 等模型设计。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | Hugging Face XDOF/ABC-130k（文章给出链接） |
| 代码 | amazon-far/abc 仓库 |

## 局限与风险

- “迄今最大”等规模表述来自论文自述，未经独立核对。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [bimanual-manipulation](../tasks/bimanual-manipulation.md)
- [teleoperation](../tasks/teleoperation.md)
- [imitation-learning](../methods/imitation-learning.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://abc.bot/)
