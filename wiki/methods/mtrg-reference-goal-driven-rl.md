---
type: method
tags: [humanoid, reinforcement-learning, motion-imitation, parkour, goal-conditioned, unitree-g1, sim2real]
status: complete
updated: 2026-06-12
related:
  - ./zest.md
  - ./hil-hybrid-imitation-learning.md
  - ./deepmimic.md
  - ../tasks/humanoid-locomotion.md
  - ../concepts/curriculum-learning.md
sources:
  - ../../sources/papers/mtrg_reference_goal_driven_rl_arxiv_2602_20375.md
summary: "MTRG 用单一 goal-conditioned 策略联合参考塑形模仿与纯目标泛化，参考仅出现在训练奖励中，在 G1 箱式跑酷上超越 ZEST tracking 与 tabula rasa 的 OOD 鲁棒性。"
---

# MTRG: Multi-Task Reference and Goal-Driven RL

**MTRG**（本库对 arXiv:2602.20375 的工作简称）把参考运动当作**行为塑形先验**而非部署时约束：一个策略只观察当前状态与 **2D 目标位置**，在训练中同时接受**稠密模仿奖励**与**稀疏目标奖励**，从而学会可复用、可转向、可应对 OOD 初始条件的人形跑酷技能。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MTRG | Multi-Task Reference and Goal-Driven RL | 参考塑形 + 目标泛化的并行多任务框架 |
| RL | Reinforcement Learning | PPO 训练 G1 全身策略 |
| MDP | Markov Decision Process | 共享观测/动作、分任务奖励的建模方式 |
| PD | Proportional-Derivative Control | 残差动作映射为关节目标再算力矩 |
| OOD | Out-of-Distribution | 训练分布外的初始位姿、距离与箱高 |
| PPO | Proximal Policy Optimization | Isaac Lab 中的 on-policy 优化器 |
| MoCap | Motion Capture | walk-jump / climb 等技能的参考来源 |

## 为什么重要

人形跑酷需要**像人**又**能改**。纯 tracking（含 [ZEST](./zest.md) 式部署时跟参考）在初始位姿偏离演示时常「硬跟参考」导致失败；纯任务 RL 动作难看。[HIL](./hil-hybrid-imitation-learning.md) 用对抗缓解但难上硬件。MTRG 用**更简单的多任务奖励分解**达到：nominal 与 beyond-nominal 成功率均优于 ZEST mocap 与 tabula rasa（论文 Table I）。

## 主要技术路线

```mermaid
flowchart TB
  subgraph inputs [策略输入 — 部署与训练一致]
    S[本体状态 s_t]
    G[目标 g_t 2D 根位置]
  end
  subgraph tasks [并行任务 — 共享 π]
    I[参考塑形模仿<br/>g_t 来自参考<br/>稠密 r_track]
    Gen[目标泛化<br/>g_t 随机<br/>稀疏 r_goal]
  end
  subgraph curriculum [λ 课程 — 来自 ZEST]
    W[虚拟辅助扳手 β(λ)]
    Mix[模仿采样概率 p_imi(λ)]
  end
  S --> PI[Goal-conditioned π]
  G --> PI
  curriculum --> I
  curriculum --> Gen
  I --> PI
  Gen --> PI
  PI --> G1[Unitree G1 箱式跑酷]
```

## 核心机制

### 1. 参考不进策略

模仿任务中参考只定义 **goal 与 tracking reward**；策略**看不到**轨迹、相位或未来姿态。泛化任务中 goal 完全随机——迫使同一 \(\pi(s,g)\) 学会「冲目标」而非「播片」。

### 2. 残差动作且无参考前馈

\(\bm{q}^{cmd}=\bar{\bm{q}}+\bm{\Sigma}\bm{a}_t\)，**不**把参考关节角作为 PD 前馈，以便泛化时偏离参考。

### 3. 与 ZEST 共享的 \(\lambda\) 课程

标量难度 \(\lambda\) 同时控制：(a) 基座辅助扳手幅度；(b) 模仿 vs 泛化任务采样比例；(c) 初始状态/目标随机化范围。对 box-climb 等高动态技能收敛关键。

### 4. 非对称 critic

Critic 见 task indicator 与接触力、辅助扳手等特权信息，仅用于 value 估计。

## 实验要点（G1）

| 技能 | 泛化行为示例 |
|------|----------------|
| walk-jump | 远则先走再跳，近则直接起跳 |
| walk-climb | 左右腿领先自适应攀爬 |
| climb-down | 单脚蹬箱调整重心再下 |

- **长程组合**：多箱 MuJoCo 序列串联三技能策略。
- **对比**：ZEST mocap 在 beyond-nominal 上 walk-jump success **0.17** vs MTRG **0.62**（论文表 I）。

## 与 HIL / ZEST 的分工

| 方法 | 场景 | 参考角色 | 对抗 | 真机 |
|------|------|----------|------|------|
| [HIL](./hil-hybrid-imitation-learning.md) | 物理角色动画 | tracking + AMP 并行 | 是 | 否 |
| [ZEST](./zest.md) | 多形态硬件模仿 | 部署时下一步参考 | 否 | 是 |
| **MTRG** | G1 箱式跑酷 | 仅训练奖励塑形 | 否 | 是（MoCap 全局位姿） |

## 常见误区

- **不是** ZEST 的简单超集——部署时**不需要**参考轨迹；与 ZEST「极简 tracking 接口」是互补路线。
- **感知**：硬件实验依赖 MoCap 全局位姿反馈；论文讨论可扩展机载外感受，但当前 box 技能未给箱体精确位姿。

## 关联页面

- [ZEST](./zest.md) — assistive wrench 课程与 tracking 基线
- [HIL](./hil-hybrid-imitation-learning.md) — 对抗式混合模仿对照
- [DeepMimic](./deepmimic.md) — 显式 tracking 传统
- [Curriculum Learning](../concepts/curriculum-learning.md)
- [Humanoid Locomotion](../tasks/humanoid-locomotion.md)
- [Unitree G1](../entities/unitree-g1.md)

## 参考来源

- [Generalizing from References using a Multi-Task Reference and Goal-Driven RL Framework](../../sources/papers/mtrg_reference_goal_driven_rl_arxiv_2602_20375.md)
- [arXiv:2602.20375](https://arxiv.org/abs/2602.20375)
- [演示视频](https://youtu.be/9NamvWhtFPM)

## 推荐继续阅读

- [ZEST 论文](https://arxiv.org/abs/2602.00401) — 辅助扳手与跨形态 tracking 细节
- [HIL 演示](https://youtu.be/le4248gIMME) — 同作者早期混合模仿与场景点云设计
