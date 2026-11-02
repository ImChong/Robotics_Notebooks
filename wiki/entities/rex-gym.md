---
type: entity
project_id: rex-gym
tags:
  - repo
  - independent
  - quadruped
  - rl
  - openai-gym
  - ppo
  - 3d-printed
status: complete
updated: 2026-11-01
code: https://github.com/nicrusso7/rex-gym
related:
  - ../methods/reinforcement-learning.md
  - ../entities/quadruped-robot.md
  - ../tasks/locomotion.md
  - ../concepts/reward-design.md
  - ../overview/quadruped-open-source-projects-technology-map.md
sources:
  - ../../sources/repos/rex_gym.md
  - ../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md
summary: "Rex-Gym 是 3D 打印开源四足机器人 Rex 的 OpenAI Gym 训练环境集合：含 Rex URDF、PPO 学习智能体与训练 / 回放脚本，目标是仿真学到的策略无需再手调即可迁移到真机。"
---

# Rex-Gym

**Rex-Gym**（[GitHub](https://github.com/nicrusso7/rex-gym)）来自 [古月居转载的四足机器人开源项目综述](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)，属「强化学习」组。

## 一句话定义

**Rex-Gym 是 3D 打印开源四足机器人 Rex 的 OpenAI Gym 训练环境集合：含 Rex URDF、PPO 学习智能体与训练 / 回放脚本，目标是仿真学到的策略无需再手调即可迁移到真机。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PPO | Proximal Policy Optimization | 近端策略优化 |
| URDF | Unified Robot Description Format | 机器人描述格式 |

## 为什么重要

- 综述把它放在「强化学习」这条技术路线下，作为可读源码的参照项目；横向对比见 [四足机器人开源项目技术地图：从极简运动学到研究级硬件（36 个 GitHub 项目）](../overview/quadruped-open-source-projects-technology-map.md)。
- 文中星标：1089（文章表述，入库日未复核）。

## 核心信息

| 项 | 内容 |
|----|------|
| **仓库** | [nicrusso7/rex-gym](https://github.com/nicrusso7/rex-gym) |
| **最近提交** | 2021-05-15 |
| **可达性** | `git ls-remote` 可达（2026-11-01） |
| **文章归类** | 强化学习 |

## 文章说法核对

- **文章：** 文章称其提供多个训练任务与精心设计的奖励函数（前进速度、能量、冲击力、姿态偏差、关节速度等惩罚），并使用课程学习。
- **README / 核查：** README 称仓库包含用于训练 Rex 的 Gym 环境、Rex URDF 模型、PPO 学习智能体与批量训练 / 策略回放 CLI；相关仓库 rexctl（控制 CLI）、rex-models（URDF 可视化）。
- **差异与保留：** 奖励函数项与课程学习在 README 前段未见，来自文章，未核实。

## 关联页面

- [reinforcement-learning](../methods/reinforcement-learning.md)
- [quadruped-robot](../entities/quadruped-robot.md)
- [locomotion](../tasks/locomotion.md)
- [reward-design](../concepts/reward-design.md)
- [quadruped-open-source-projects-technology-map](../overview/quadruped-open-source-projects-technology-map.md)

## 参考来源

- [rex_gym.md](../../sources/repos/rex_gym.md)
- [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)
