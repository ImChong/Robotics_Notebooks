---
type: entity
tags:
  - dataset
  - sim
  - sim2real
  - mobile-manipulation
status: draft
updated: 2026-10-20
summary: "MolmoBot-Engine 在仿真住宅中生成的 180 万条专家轨迹，主打零样本仿真到真实迁移。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/embodied-scaling-laws.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# MolmoBot-Data

## 一句话定义

**MolmoBot-Data**：在 MolmoSpaces 仿真住宅中程序化生成的铰接物体操作与抓取放置专家轨迹。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真到真实迁移 |
| RB-Y1 | Rainbow Robotics 移动双臂平台（名称） | 文章中的机器人型号 |
| zero-shot | 零样本 | 未经真机微调的迁移 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | Allen Institute for AI（Ai2） |
| 时间 | 2026-03 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 180 万条专家轨迹 |
| 平台 | Franka FR3 桌面操作；Rainbow Robotics RB-Y1 移动操作 |
| 任务 | 8 种任务配置（如开门、开抽屉、按颜色/位置放置） |

## 为什么重要

- 论文称仅靠足够大且多样的仿真数据即可零样本迁移到真实环境，是 Sim2Real 数据路线的代表。

## 核心原理

MolmoBot-Engine 程序化生成轨迹，覆盖静态与移动两类操作。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | Hugging Face allenai/MolmoBot-Data（文章给出链接） |

## 局限与风险

- 零样本迁移结论依赖论文实验，任务范围较窄（8 种配置）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [embodied-scaling-laws](../concepts/embodied-scaling-laws.md)
- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://huggingface.co/datasets/allenai/MolmoBot-Data)
