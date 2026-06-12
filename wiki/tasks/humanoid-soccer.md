---
type: task
tags: [humanoid, soccer, robocup, rl, perception, kicking]
status: drafting
updated: 2026-06-12
related:
  - ./locomotion.md
  - ../concepts/humanoid-multi-robot-coordination.md
  - ../methods/reinforcement-learning.md
  - ../methods/imitation-learning.md
  - ../methods/marl.md
  - ../methods/paid-framework.md
  - ../methods/htwk-gym.md
  - ../entities/booster-robocup-demo.md
  - ../entities/unitree-g1.md
  - ../entities/paper-robonaldo-humanoid-soccer-shooting.md
  - ../entities/paper-humanoid-soccer-swarm-intelligence.md
  - ../entities/paper-notebook-a-hierarchical-model-based-system-for-high-perfo.md
  - ../entities/paper-notebook-learning-soccer-skills-for-humanoid-robots.md
sources:
  - ../../sources/repos/htwk_gym.md
  - ../../sources/repos/humanoid_soccer.md
  - ../../sources/repos/booster-robocup-demo.md
  - ../../sources/papers/robonaldo_arxiv_2606_11092.md
  - ../../sources/papers/humanoid_soccer_swarm_intelligence_sensors_2025.md
  - ../../sources/papers/artemis_humanoid_soccer_team_coordination_arxiv_2512_09431.md
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

### 5. 多机协作与群控
竞技足球要求 **角色分配、站位/编队、传球线路与防守补位**，且 RoboCup 联赛 **通信带宽持续收紧**（SPL 队均 WiFi 包量近年大幅下降）。群控与单机「追球射门」技能正交：前者回答 **谁去踢、谁留守、如何避开工友扎堆**。详见 [人形多机协调](../concepts/humanoid-multi-robot-coordination.md)。

## 主要技术路线

### 强化学习 (RL) 驱动
通过在大规模并行仿真（如 Isaac Gym/Lab）中训练，直接获取端到端的运动与技能。
- **HTWK-Gym**：针对 Booster T1/K1 平台的足球任务优化框架。
- **PAiD (Perception-Action Integrated Decision-making)**：将感知与动作解耦并渐进式融合，实现更稳健的踢球。
- **[RoboNaldo](../entities/paper-robonaldo-humanoid-soccer-shooting.md)**：以单条人类踢球参考为 scaffold 的 **三阶段 motion-guided curriculum RL**，在 G1 上实现 **亚米级点瞄准射门**、**13 m/s 级触球球速** 与 **来球 one-touch** 室外真机演示。

### 分层状态机 + 技能库
将比赛逻辑划分为多个状态（寻球、追球、对齐、踢球），每个状态对应一个底层控制器。
- 代表项目：[Booster RoboCup Demo](../entities/booster-robocup-demo.md)

### 多机战术 / 群控
| 范式 | 代表一手资料 | 要点 |
|------|--------------|------|
| **集中式行为管理** | [ARTEMIS](../entities/paper-notebook-a-hierarchical-model-based-system-for-high-perfo.md)（arXiv:2512.09431） | 立体视觉检测队友/对手 + behavior planner 角色与射门；**2024 Adult-Size 冠军** |
| **Swarm 去中心化** | [Swarm Intelligence 人形足球](../entities/paper-humanoid-soccer-swarm-intelligence.md)（Sensors 2025） | UDP + ACO 角色 + flocking 编队；Webots 4v4 相对集中式 **进球 +25–40%** |
| **极低带宽拍卖（SPL）** | [arXiv:2401.15026](../../sources/papers/robocup_spl_limited_communication_coordination_arxiv_2401_15026.md) | DWM 预测 + 市场拍卖 + Voronoi；**NAO 真机 RoboCup** |

## 关键技能 (Skills)

| 技能 | 技术难点 | 典型实现 |
|------|----------|---------|
| **寻球 (Search)** | 广域视觉扫描、头部关节协同 | YOLOv8 + 分级搜索策略 |
| **接近与对齐 (Chase & Align)** | 全向步态、动态 ZMP 调节 | 参数化行走 (htwk-gym) |
| **踢球 (Kick)** | 单脚支撑平衡、摆腿轨迹规划、高冲量触球时机 | RLVisionKick / PAiD / RoboNaldo |
| **跌倒恢复 (Get up)** | 接触力反馈、全身协同规划 | 预设 Keyframe / RL Getup |

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Isaac Gym | NVIDIA Isaac Gym | GPU 并行刚体仿真训练环境 |
| ZMP | Zero Moment Point | 足式平衡判据，地面反力合力矩为零的点 |
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |
| ACO | Ant Colony Optimization | 蚁群优化；人形足球 swarm 角色分配常用机制 |
| SPL | Standard Platform League | RoboCup NAO 人形联赛；通信配额极严 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 参考来源

- [NaoHTWK/htwk-gym 源码仓库](../../sources/repos/htwk_gym.md) — 针对 Booster T1/K1 的足球 RL 框架
- [TeleHuman/HumanoidSoccer (PAiD) 源码仓库](../../sources/repos/humanoid_soccer.md) — 针对 Unitree G1 的渐进式足球学习
- [robonaldo_arxiv_2606_11092.md](../../sources/papers/robonaldo_arxiv_2606_11092.md) — RoboNaldo 人形射门课程 RL 与 G1 机载感知摘录
- [Booster Robotics RoboCup Demo](../../wiki/entities/booster-robocup-demo.md) — 完整的足球比赛软件方案
- [humanoid_soccer_swarm_intelligence_sensors_2025.md](../../sources/papers/humanoid_soccer_swarm_intelligence_sensors_2025.md) — 人形足球 swarm 群控（Sensors 2025）
- [artemis_humanoid_soccer_team_coordination_arxiv_2512_09431.md](../../sources/papers/artemis_humanoid_soccer_team_coordination_arxiv_2512_09431.md) — ARTEMIS 冠军系统群控摘录

## 关联系统/方法

- [Locomotion](./locomotion.md) — 足球任务的基础
- [PAiD Framework](../methods/paid-framework.md) — 渐进式感知动作学习
- [人形足球技能学习方法选型指南](../queries/humanoid-soccer-skill-learning-method-selection.md) — PAiD vs RoboNaldo 选型
- [RoboNaldo](../entities/paper-robonaldo-humanoid-soccer-shooting.md) — 点级瞄准与高冲量射门课程 RL
- [HTWK-Gym](../methods/htwk-gym.md) — 足球专项 RL 训练环境
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [人形多机协调](../concepts/humanoid-multi-robot-coordination.md) — 群控范式总览
- [Swarm Intelligence 人形足球](../entities/paper-humanoid-soccer-swarm-intelligence.md) — ACO+flocking 去中心化
- [ARTEMIS 人形足球系统](../entities/paper-notebook-a-hierarchical-model-based-system-for-high-perfo.md) — 集中式战术层真机冠军
- [MARL](../methods/marl.md) — 学习式多体协调
