---
type: entity
tags:
  - dataset
  - sim
  - synthetic
  - pretraining
status: draft
updated: 2026-10-20
summary: "高保真合成预训练数据：超 63 万条轨迹、7,433 小时，覆盖 4 种本体与 70 项任务。"
related:
  - ../overview/embodied-dataset-panorama-2026-10.md
  - ../concepts/embodied-scaling-laws.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md
---

# InternData-A1

## 一句话定义

**InternData-A1**：用自动化、解耦的组合式仿真流程生成的合成操作数据，面向通用策略预训练。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| π0 | pi-zero（对比基线） | 文章用其作为对照模型 |
| Sim2Real | Simulation to Real | 仿真到真实迁移 |
| VLA | Vision-Language-Action | 视觉-语言-动作模型 |

## 数据集速查

| 维度 | 速查 |
|------|------|
| 机构 | 上海人工智能实验室 |
| 时间 | 2025-11 |
| 许可证 | 文章未注明，以官方页为准 |
| 重定向就绪度 | 未评估：需按目标机器人形态核对动作与观测空间 |
| 规模 | 超 63 万条轨迹；7,433 小时 |
| 覆盖 | 4 种本体、18 类技能、70 项任务、227 个场景 |
| 对象 | 刚体、铰接、柔性物体、流体 |

## 为什么重要

- 覆盖刚体、铰接、柔性物体与流体，并支持长程技能组合，是合成数据做预训练的规模化样例。

## 核心原理

组合式仿真流水线自动生成长程技能组合与异构本体数据。

## 工程实践

| 项 | 说明 |
|----|------|
| 获取 | Hugging Face InternRobotics/InternData-A1（文章给出链接） |

## 局限与风险

- “与 π0 相当”为论文自述，仿真占比高，真实迁移效果需独立验证（推测）。
- 本页事实主要转述自 2026-10-19 公众号综述，尚未逐条对照官方页；使用前以官方说明为准。

## 关联页面

- [具身数据集全景（85 项去重映射）](../overview/embodied-dataset-panorama-2026-10.md)
- [embodied-scaling-laws](../concepts/embodied-scaling-laws.md)
- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)

## 参考来源

- [2026-10-19 公众号综述：全球 85 个具身智能数据集全景](../../sources/datasets/wechat-embodied-dataset-panorama-2026-10-19.md)

## 推荐继续阅读

- [官方数据页](https://internrobotics.github.io/interndata-a1.github.io/)
