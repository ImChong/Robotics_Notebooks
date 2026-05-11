---
type: task
tags: [locomotion, bipedal, humanoid, rl, control]
status: complete
updated: 2026-04-20
related:
  - ../concepts/whole-body-control.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../methods/imitation-learning.md
  - ../concepts/capture-point-dcm.md
  - ../concepts/gait-generation.md
  - ../concepts/footstep-planning.md
  - ../entities/unitree.md
  - ./ultra-survey.md
  - ./manipulation.md
  - ./loco-manipulation.md
  - ./balance-recovery.md
  - ../queries/humanoid-hardware-selection.md
  - ../queries/humanoid-rl-cookbook.md
  - ../concepts/wheel-legged-quadruped.md
  - ../entities/quadruped-robot.md
sources:
  - ../../sources/papers/policy_optimization.md
  - ../../sources/papers/state_estimation.md
summary: "Locomotion 研究机器人如何稳定、高效地在不同地形上移动，是腿式与人形控制的核心任务页。"
---

# Locomotion

**运动/行走**：让机器人（尤其人形/足式）实现稳定、高效、多地形移动的能力。

## 一句话定义

让机器人在不需要轮子的情况下，用腿走路，而且走得稳、走得快、走得自然。

## 核心挑战

### 1. 平衡
人形机器人是天然不稳定的系统，必须主动维持平衡。

- 静态平衡：重心在支撑多边形内
- 动态平衡：ZMP（Zero Moment Point）条件
- 接触力分配：多接触时的力分配问题

### 2. 接触切换
行走本质是不断在单脚支撑和双脚支撑之间切换，每次切换都容易失稳。

### 3. 高维动作空间
30+ 自由度，每次决策都要协调所有关节。

### 4. 地形变化
平坦、崎岖、不平整、楼梯——每种地形需要不同的步态策略。

## 主要方法路线

### 传统控制路线
- **ZMP + 预观控制**：经典人形行走（Honda ASIMO）
- **LIP + 步长调节**：简单高效的行走控制
- **Hybrid Zero Dynamics**：考虑机器人动力学结构的步态生成

### 学习路线
- **RL from scratch**：直接在仿真里训，不需要人工步态设计。
  - 代表：PPO 训四足/双足行走（Legged Gym, IsaacGymEnvs）。
  - 新趋势：**BRRL / BPO (2026)** 在 IsaacLab 环境下报告了比 PPO 更稳健的 locomotion 训练表现。
- **IL + RL**：用 MoCap 数据初始化，再用 RL 提升。
  - 代表：DeepMimic, AMP
- **Multi-Gait Learning (多步态学习)**：在一个统一的 RL 框架下训练多种步态。
  - 新趋势：使用 **Selective AMP (选择性 AMP)** 策略，对周期性步态（如行走、上楼梯）应用 AMP 以提高稳定性，对高动态步态（如跑、跳）则省略 AMP，避免正则化过度约束。
- **世界模型**：学习环境模型，在模型里规划。
  - 代表：Dreamer, LIFT

## 评价指标

- **行走速度**：m/s
- **能耗效率**：J/kg/m 或 Cost of Transport (CoT)
- **地形适应能力**：是否能处理楼梯、不平整地面
- **稳定性**：摔倒频率
- **运动自然性**：和人类步态的相似度
- **泛化能力**：能否迁移到未见过的地形

## 参考来源

- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning* — legged_gym 路线奠基论文
- Peng et al., *DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills* — IL+RL 融合代表
- [Locomotion RL 论文导航](../../references/papers/locomotion-rl.md) — 论文集合
- **ingest 档案：** [sources/papers/policy_optimization.md](../../sources/papers/policy_optimization.md) — PPO/SAC/TD3 + Rudin legged_gym
- **ingest 档案：** [sources/papers/state_estimation.md](../../sources/papers/state_estimation.md) — EKF/InEKF 状态估计
- **ingest 档案：** [Multi-Gait Learning for Humanoid Robots Using Reinforcement Learning with Selective Adversarial Motion Priority](../../sources/papers/multi-gait-learning.md) — 多步态学习中的 Selective AMP 策略

## 关联系统/方法

- [Whole-Body Control](../concepts/whole-body-control.md)
- [Sim2Real](../concepts/sim2real.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [MPC](https://en.wikipedia.org/wiki/Model_predictive_control)（在行走中常用于步态预览）
- [Capture Point / DCM](../concepts/capture-point-dcm.md)（步行平衡与扰动恢复的核心方法）
- [Lyapunov 稳定性](../formalizations/lyapunov.md)（分析闭环步态稳定性、扰动恢复和误差收敛的统一语言）
- [Gait Generation](../concepts/gait-generation.md)（步态时序编排：CPG / 参数化 / MPC 联合优化）
- [Footstep Planning](../concepts/footstep-planning.md)（接触序列规划：每步踩哪里、踩多久）
- [Terrain Adaptation](../concepts/terrain-adaptation.md)（把高度图 / 点云转成可执行的落脚点与姿态调整）
- [轮足四足机器人（四轮足）](../concepts/wheel-legged-quadruped.md)（Go2W / B2W 类：腿末驱动轮与足式步态混合）
- [HiPAN](../methods/hipan.md)（四足在非结构化 3D 环境中的分层深度导航 + 姿态自适应低层跟踪）
- [四足机器人](../entities/quadruped-robot.md)（四足形态与典型平台的实体入口）
- [Unitree](../entities/unitree.md)（当前主流人形/四足研究硬件平台）
- [ULTRA：统一多模态 loco-manipulation 控制](./ultra-survey.md)（UIUC 2026，新一代全身移动操作统一控制器）
- [Query：何时用 WBC vs RL？](../queries/when-to-use-wbc-vs-rl.md) — 实践决策指南

## 关联任务

- [Manipulation](./manipulation.md)：行走+操作 = loco-manipulation
- [Loco-Manipulation](./loco-manipulation.md)：全身移动操作的统一挑战
- [Balance Recovery](./balance-recovery.md)：扰动恢复，鲁棒 locomotion 的核心子能力
- [Query：人形机器人运动控制 Know-How](../queries/humanoid-motion-control-know-how.md) — locomotion 实战经验结构化摘要
- [Query：开源运动控制项目导航](../queries/open-source-motion-control-projects.md) — 主流开源框架与项目概览

## 继续深挖入口

如果你想沿着 locomotion 继续往下挖，建议从这里进入：

### 论文入口
- [Locomotion RL 论文导航](../../references/papers/locomotion-rl.md)

### Benchmark 入口
- [Locomotion Benchmarks](../../references/benchmarks/locomotion-benchmarks.md)

### 开源项目 / 框架入口
- [RL Frameworks](../../references/repos/rl-frameworks.md)
- [Simulation](../../references/repos/simulation.md)

## 关联页面

- [Humanoid Locomotion](./humanoid-locomotion.md) — 人形机器人全身移动任务
- [Hybrid Locomotion](./hybrid-locomotion.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [MPC](../methods/model-predictive-control.md)

## 推荐继续阅读

- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*（legged_gym 原论文）
- Won et al., *Perpetual Robot Control: Designing Robot Agility and Recovery*（CPI + RL 路线）
- Jin et al., *Rapid and Scalable Reinforcement Learning for Legged Robots*（Isaac Lab 路线）
- [Humanoid Control Roadmap](../roadmaps/humanoid-control-roadmap.md) — 人形机器人运控的学习成长路线
- [Query：人形机器人硬件怎么选](../queries/humanoid-hardware-selection.md)
- [Query：人形机器人 RL 实战 Cookbook](../queries/humanoid-rl-cookbook.md)
