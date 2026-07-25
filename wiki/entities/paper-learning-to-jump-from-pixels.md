---
type: entity
tags: [paper, rl, vision, locomotion, quadruped, sim2real, mit, jumping]
status: complete
updated: 2026-07-25
arxiv: "2110.15344"
related:
  - ./mit-mini-cheetah.md
  - ./paper-rapid-locomotion-rl.md
  - ./paper-vision-aided-dynamic-exploration-mini-cheetah.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/learning_to_jump_from_pixels_arxiv_2110_15344.md
  - ../../sources/sites/jumping-from-pixels.md
summary: "Margolis et al. arXiv:2110.15344：像素级视觉引导 Mini Cheetah 在间断地形（沟/障）上规划并执行敏捷跳跃。"
---

# Learning to Jump from Pixels

## 一句话定义

**Margolis et al.（MIT，[arXiv:2110.15344](https://arxiv.org/abs/2110.15344)）** 学习从**像素**出发，在**间断地形**（沟、障碍）上前瞻规划并执行**敏捷跳跃**；强调剧烈运动导致机载相机晃动时的实时视觉挑战，平台为 **Mini Cheetah**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 跳跃策略学习 |
| RGB | Red-Green-Blue | 像素观测模态 |
| Sim2Real | Simulation to Real | 仿真到真机 |
| FoV | Field of View | 跳跃时视野剧烈变化 |
| MPC | Model Predictive Control | 对照的模型基路径语境 |

## 为什么重要

- 补齐「连续粗糙可盲走」之外的**间断地形**能力。
- 把视觉抖动/运动模糊变成一等公民问题，而不是假设完美深度图。
- 与同组 [Rapid Locomotion RL](./paper-rapid-locomotion-rl.md) 共同展示 Mini Cheetah 学习线。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 麻省理工（MIT）CSAIL / Biomimetics |
| **项目页** | https://sites.google.com/view/jumpingfrompixels |
| **开源** | **未开源**（项目页核查无钉死训练仓） |

## 核心原理

- 连续地形：鲁棒行走为主；间断地形：必须视觉前瞻 + 超出稳健行走的跳跃技能。
- 动态跳跃 → 传感器大幅运动 → 实时视觉管线需抗晃动。
- 学习框架连接像素观测与跳跃动作。

```mermaid
flowchart LR
  img["机载像素"] --> vis["视觉编码 / 地形线索"]
  vis --> pol["跳跃策略"]
  pol --> act["关节 / 基座命令"]
  act --> bot["Mini Cheetah"]
```

## 源码运行时序图

**不适用**（项目页与论文未提供可运行官方代码入口）。

## 工程实践

| 项 | 建议 |
|----|------|
| 数据 | 仿真中覆盖沟宽/障高课程 |
| 视觉 | 增强运动模糊与曝光变化随机化 |
| 安全 | 真机跳跃必须防护与逐步增沟宽 |

## 评测

| 维度 | 要点 |
|------|------|
| 地形 | 沟与障碍等间断设置 |
| 感知 | 像素级输入 |
| 平台 | Mini Cheetah |

## 结论

**总判：** 本文是 Mini Cheetah **视觉敏捷技能**代表作——从「看得到」推进到「跳得过」。

- 真影响：间断地形 + 像素策略 + 运动中视觉。
- 次要代价：未开源；复现门槛高。
- 部署：作方法参考；工程上可与深度落脚模块混合。

## 局限与风险

- 无代码仓。
- 跳跃失败对硬件冲击大。

## 关联页面

- [Rapid Locomotion RL](./paper-rapid-locomotion-rl.md)
- [Vision-aided exploration](./paper-vision-aided-dynamic-exploration-mini-cheetah.md)
- [Sim2Real](../concepts/sim2real.md)
- [MIT Mini Cheetah](./mit-mini-cheetah.md)

## 参考来源

- [论文归档](../../sources/papers/learning_to_jump_from_pixels_arxiv_2110_15344.md)
- [项目页归档](../../sources/sites/jumping-from-pixels.md)

## 推荐继续阅读

- arXiv：<https://arxiv.org/abs/2110.15344>
- 项目页：<https://sites.google.com/view/jumpingfrompixels>
