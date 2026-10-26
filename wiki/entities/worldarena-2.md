---
type: entity
tags: [benchmark, embodied-ai, world-models, visuotactile, reinforcement-learning, sim-to-real, xiaomi-robotics]
status: complete
updated: 2026-10-26
project: https://huggingface.co/spaces/WorldArena/WorldArena2.0
code: https://github.com/WorldArena2/WorldArena-2.0
related:
  - ./worldarena.md
  - ./paper-pace-tactile-conditioning.md
  - ./paper-foretac.md
  - ../methods/generative-world-models.md
sources:
  - ../../sources/repos/worldarena-2.md
summary: "WorldArena 2.0 将具身世界模型评测扩展到视触觉、在线强化学习环境和跨本体仿真到真机；官方排行榜托管于 Hugging Face。"
---

# WorldArena 2.0：具身世界模型的多模态、功能与平台扩展评测

**WorldArena 2.0** 是 WorldArena 具身世界模型基准的下一代版本。它把评测从视觉视频质量与离线功能扩展到三条轴线：

| 扩展方向 | 评测重点 |
|---|---|
| 模态 | 从纯视觉扩展到联合视觉–触觉的世界模型评估 |
| 功能 | 将世界模型作为在线强化学习交互环境，检验其能否稳定支持策略优化 |
| 平台 | 从仿真扩展到真实机器人系统，评估跨本体 sim-to-real 能力 |

## 与 WorldArena 1.0 的关系

[WorldArena 1.0](./worldarena.md) 已把视频预测、数据引擎、策略评估和动作规划纳入统一框架。2.0 保留统一评测目标，新增视触觉、在线 RL 和真机迁移能力；因此 1.0 与 2.0 的结果应按各自协议解读，不能将分数直接混排。

## 入口与状态

- 官方实现仓库：<https://github.com/WorldArena2/WorldArena-2.0>
- 官方排行榜：<https://huggingface.co/spaces/WorldArena/WorldArena2.0>
- 仓库 README 将排行榜称为官方 WorldArena leaderboard。
- Hugging Face 结果说明中列有 Xiaomi Robotics 团队条目；具体赛道排名应以排行榜当前结果表为准。

## 来源

- [WorldArena 2.0 官方仓库](../../sources/repos/worldarena-2.md)
- [WorldArena 1.0](./worldarena.md)
