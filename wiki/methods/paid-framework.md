---
type: method
tags: [rl, imitation-learning, perception, soccer, humanoid, unitree-g1]
status: complete
updated: 2026-11-22
related:
  - ../queries/robot-perception-stack-selection-loop.md
  - ../tasks/humanoid-soccer.md
  - ../entities/unitree-g1.md
  - ../entities/paper-robonaldo-humanoid-soccer-shooting.md
  - ../entities/paper-notebook-learning-soccer-skills-for-humanoid-robots.md
  - ./reinforcement-learning.md
  - ./imitation-learning.md
sources:
  - ../../sources/repos/humanoid_soccer.md
  - ../../sources/papers/robonaldo_arxiv_2606_11092.md
  - ../../sources/repos/robonaldo.md
summary: "PAiD (Perception-Action integrated Decision-making) 是一种渐进式的人形机器人技能学习框架，通过模仿学习与感知-动作融合实现鲁棒的类人化踢球。"
original:
  - ../../sources/papers/humanoid_pnb_learning-soccer-skills-for-humanoid-robots.md
  - ../../sources/repos/humanoid_soccer.md
---

# PAiD Framework

**PAiD (Perception-Action integrated Decision-making)** 是由 TeleHuman 研究团队提出的一种针对人形机器人足球技能的渐进式学习框架。其核心思想是将复杂的感知-动作任务分解为多个阶段，逐步增加系统复杂度，以解决端到端学习中常见的收敛难与泛化差的问题。

## 三阶段渐进式训练流程

PAiD 的成功关键在于其结构化的训练方案：

### 1. 动作技能习得 (Skill Acquisition)
- **目标**：获取基础的踢球肢体协调能力。
- **方法**：使用**模仿学习 (Imitation Learning)**。从人类动作捕捉数据中提取关键特征，通过 RL 训练机器人跟踪这些动作。
- **输出**：一个能够执行自然摆腿动作的基准策略（Motion Policy）。

### 2. 感知集成 (Perception Integration)
- **目标**：赋予机器人“看球”并调整动作的能力。
- **方法**：将视觉感知特征（如球的位置、速度）引入状态空间。训练策略根据感知输入动态调整第一阶段习得的动作。
- **输出**：一个能够在不同足球初始位姿下完成踢球的感知-动作策略。

### 3. 物理感知 Sim-to-Real (Physics-aware Transfer)
- **目标**：确保策略在真实物理环境（如 Unitree G1）中的有效性。
- **方法**：
    - **领域随机化**：对质量、摩擦力、视觉噪声进行大规模扰动。
    - **时延模拟**：在仿真中引入控制环路的时延，增强鲁棒性。
- **输出**：可直接部署到 Unitree G1 上的部署策略。

## 核心原理

采用**阶段化策略延续与状态空间渐进扩展**：Stage I 基于 GMR 重定向的 13 条踢球参考以自适应跟踪收敛稳定动作底座；Stage II 从 Stage I 检查点恢复，将骨盆系球位置及门位置引入状态并通过 RNN/LSTM 聚合滚动球时序运动信息；Stage III 叠加球接触 CMA-ES 动力学对齐与控制延迟模拟实现 G1 真机室内外鲁棒执行。

## 主要技术路线

| 阶段 | 核心方法 | 关键技术 |
|------|---------|---------|
| **1. 动作习得** | 模仿学习 (IL) | 基于 MoCap 数据的 [Behavior Cloning](../formalizations/behavior-cloning-loss.md) |
| **2. 感知集成** | 强化学习 (RL) | 紧耦合视觉特征的 PPO 算法 |
| **3. 真机迁移** | Sim-to-Real | [领域随机化](../concepts/domain-randomization.md) + 时延反馈补偿 |

## 技术特色

- **类人化 (Human-like)**：通过第一阶段的约束，机器人表现出更符合人类生物力学的踢球动作，而非仅仅是简单的肢体碰撞。
- **RNN 网络架构**：使用循环神经网络处理感知序列，能够更好地利用历史信息应对视觉遮挡或足球滚动。
- **高泛化性**：支持多种地形（室内地板、室外草坪）以及静止/滚动的足球。

## 与 RoboNaldo 的对照（2026）

同为 **G1 + 三阶段渐进 RL** 的人形射门路线，[RoboNaldo](../entities/paper-robonaldo-humanoid-soccer-shooting.md)（arXiv:2606.11092）与 PAiD 侧重点不同：

| 维度 | PAiD | RoboNaldo |
|------|------|-----------|
| **阶段语义** | 动作习得 → 感知融合 → Sim2Real | motion tracking → 任意球适应 → 来球时机 |
| **瞄准目标** | goal-region / 成功率导向 | **点级误差**（3 m 平均 **0.73 m**） |
| **功率** | 未主打触球球速叙事 | 触球后最高 **13.10 m/s** |
| **感知** | 第一视角 RGB 融合 | **LiDAR 近距 + IR 远距** 球定位 |
| **开源（2026-07）** | 见 [HumanoidSoccer 仓](../../sources/repos/humanoid_soccer.md) | **已开源** [RoboNaldo](https://github.com/OpenDriveLab/RoboNaldo) + [Deploy](https://github.com/OpenDriveLab/RoboNaldo_Deploy) |

二者可并读：PAiD 代表 **感知–动作渐进融合** 范式；RoboNaldo 代表 **motion scaffold + 高冲量课程** 范式。

## 应用案例

在 **Unitree G1** 机器人上的实验证明，PAiD 框架能够实现：
- 高达 90% 以上的稳健踢球成功率。
- 动态调整踢球方向和力度以射门或传球。
- 在受到轻微外部扰动时仍能维持平衡并完成动作。

## 工程实践

- **开源训练一键渐进启动**：仓库通过 `bash shell/progressive_soccer_train_play.sh` 先跑 Stage I motion tracking，再自动承接恢复权重训练滚动球阶段（`Tracking-Flat-G1-SoccerMoving-RNN-v0`），部署前可借 `exp/` 下 MuJoCo 回放检查。
- **依赖版本与数据要求**：严格基于 Isaac Lab v2.1.1；室内地板到室外草坪迁移时注意调整球摩擦与观测丢帧噪声范围。

## 局限与风险

- **指标侧重成功率而非高冲量极值**：与 RoboNaldo 相比，PAiD 更强调不同方位与滚动来球踢球成功率（静球仿真 91.3%、滚球 71.9%），未突出 13 m/s 级出球极速与精确点射门，也不包含对手阻挡博弈。

## 一句话定义

**PAiD（Perception-Action integrated Decision-making）**：由 TeleHuman 团队提出的三阶段人形足球技能学习框架——先用人类踢球 MoCap 习得拟人摆腿先验，再加入第一视角位置/速度特征进行 RNN 感知融合，最后经接触与控制延迟领域随机化部署到 Unitree G1。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |
| IL | Imitation Learning | 从专家演示学习策略，奖励难定义时的主路线 |
| MoCap | Motion Capture | 动作捕捉，参考动作与演示数据的主要来源 |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |

## 为什么重要

- **打破敏捷踢球中的拟人性与感知闭环矛盾**：跳过单纯靠黑盒奖励“暴力甩腿”的非类人步态，保留生物力学自然度同时具备响应静止和滚动球的能力。
- **给出完整的公开复现入口与对比锚点**：仓库公开了运动数据 `motions`、Isaac Lab 环境、RSL-RL progressive 脚本与 MuJoCo sim2sim 接口，方便与同平台 RoboNaldo 做对照开发。

## 参考来源

- [HumanoidSoccer (PAiD) 源码仓库](../../sources/repos/humanoid_soccer.md)
- [Learning Soccer Skills for Humanoid Robots（Paper Notebooks 摘录）](../../sources/papers/humanoid_pnb_learning-soccer-skills-for-humanoid-robots.md)
- [robonaldo_arxiv_2606_11092.md](../../sources/papers/robonaldo_arxiv_2606_11092.md) — 并发射门工作对照摘录

## 推荐继续阅读

- 开源代码仓库：<https://github.com/TeleHuman/HumanoidSoccer>
- 论文实体详情：[PAiD：Learning Soccer Skills for Humanoid Robots](../entities/paper-notebook-learning-soccer-skills-for-humanoid-robots.md)

## 关联页面

- [Humanoid Soccer](../tasks/humanoid-soccer.md)
- [人形足球技能学习方法选型指南](../queries/humanoid-soccer-skill-learning-method-selection.md)
- [RoboNaldo](../entities/paper-robonaldo-humanoid-soccer-shooting.md)
- [Imitation Learning](./imitation-learning.md)
- [Reinforcement Learning](./reinforcement-learning.md)
- [Domain Randomization](../concepts/domain-randomization.md)
- [Behavior Cloning Loss](../formalizations/behavior-cloning-loss.md)
- [Unitree G1](../entities/unitree-g1.md)
