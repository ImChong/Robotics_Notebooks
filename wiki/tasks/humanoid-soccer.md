---
type: task
tags: [humanoid, soccer, robocup, rl, perception, kicking]
status: drafting
updated: 2026-04-27
related:
  - ./locomotion.md
  - ../methods/reinforcement-learning.md
  - ../methods/imitation-learning.md
  - ../methods/paid-framework.md
  - ../methods/htwk-gym.md
  - ../entities/booster-robocup-demo.md
  - ../entities/unitree-g1.md
sources:
  - ../../sources/repos/htwk_gym.md
  - ../../sources/repos/humanoid_soccer.md
  - ../../sources/repos/booster-robocup-demo.md
summary: "Humanoid Soccer 是机器人学中最具挑战性的综合任务之一，要求人形机器人集成高速行走、动态视觉、精准踢球与多机协作。"
---

# Humanoid Soccer

**人形机器人足球**：人形机器人在动态、竞争性环境下的综合表现。作为 RoboCup 的核心项目，它被认为是衡量人形机器人自主能力的重要基准。

## 一句话定义

让两队人形机器人在足球场上像人类一样踢球，不摔倒、能进球、懂配合。

## 核心挑战

### 1. 动态感知与定位
机器人必须在快速移动中识别高速滚动的足球、对手机器人、场地边线和球门，并实时更新自身位姿。

### 2. 闭环踢球 (Closed-loop Kicking)
不同于预设轨迹的踢球，竞技环境要求：
- **实时修正**：在接近球的过程中根据球的微小移动动态调整步态与出脚位置。
- **力度控制**：根据目标距离与角度调整踢球强度。

### 3. 参数化全向行走 (Omni-directional Walking)
需要具备在任意时刻改变速度、频率和航向的能力，以便进行防守、截球和过人。

### 4. 复杂环境稳定性
真实的草地（或人工草坪）往往不平整，且存在碰撞干扰，对 locomotion 的鲁棒性要求极高。

## 主要技术路线

### 强化学习 (RL) 驱动
通过在大规模并行仿真（如 Isaac Gym/Lab）中训练，直接获取端到端的运动与技能。
- **HTWK-Gym**：针对 Booster T1/K1 平台的足球任务优化框架。
- **PAiD (Perception-Action Integrated Decision-making)**：将感知与动作解耦并渐进式融合，实现更稳健的踢球。

### 分层状态机 + 技能库
将比赛逻辑划分为多个状态（寻球、追球、对齐、踢球），每个状态对应一个底层控制器。
- 代表项目：[Booster RoboCup Demo](../entities/booster-robocup-demo.md)

## 关键技能 (Skills)

| 技能 | 技术难点 | 典型实现 |
|------|----------|---------|
| **寻球 (Search)** | 广域视觉扫描、头部关节协同 | YOLOv8 + 分级搜索策略 |
| **接近与对齐 (Chase & Align)** | 全向步态、动态 ZMP 调节 | 参数化行走 (htwk-gym) |
| **踢球 (Kick)** | 单脚支撑平衡、摆腿轨迹规划 | RLVisionKick / PAiD |
| **跌倒恢复 (Get up)** | 接触力反馈、全身协同规划 | 预设 Keyframe / RL Getup |

## 参考来源

- [NaoHTWK/htwk-gym 源码仓库](../../sources/repos/htwk_gym.md) — 针对 Booster T1/K1 的足球 RL 框架
- [TeleHuman/HumanoidSoccer (PAiD) 源码仓库](../../sources/repos/humanoid_soccer.md) — 针对 Unitree G1 的渐进式足球学习
- [Booster Robotics RoboCup Demo](../../wiki/entities/booster-robocup-demo.md) — 完整的足球比赛软件方案

## 关联系统/方法

- [Locomotion](./locomotion.md) — 足球任务的基础
- [PAiD Framework](../methods/paid-framework.md) — 渐进式感知动作学习
- [HTWK-Gym](../methods/htwk-gym.md) — 足球专项 RL 训练环境
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Imitation Learning](../methods/imitation-learning.md)
