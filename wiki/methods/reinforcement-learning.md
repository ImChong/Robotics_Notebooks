---
type: method
tags: [rl, locomotion, policy-optimization, model-free, model-based]
status: complete
---

# Reinforcement Learning (RL)

**强化学习**：通过与环境交互，以最大化累积 reward 为目标学习决策策略的机器学习范式。

## 一句话定义

不需要告诉机器人“怎么做”，只需要告诉它“做得好不好”，让它自己摸索。

## 核心框架：MDP

强化学习问题通常建模为马尔可夫决策过程（MDP）：

- **状态** $s$：机器人当前感知到的环境信息
- **动作** $a$：机器人可以采取的行动
- **奖励** $r$：环境给机器人的反馈信号
- **策略** $\pi(a|s)$：在每个状态下选择动作的规则
- **折扣因子** $\gamma$：未来奖励的重要性

目标：找到最优策略 $\pi^*$ 最大化期望累积折扣奖励。

## 主要分类

### 无模型（Model-Free）
不学习环境模型，直接从交互数据学习策略。

代表算法：
- **Policy Gradient**：直接优化策略（REINFORCE, PPO, AWR）
- **Q-Learning**：学习状态-动作价值函数（DQN, DDPG, SAC）
- **Actor-Critic**：结合两者（PPO, SAC, TD3）

### 有模型（Model-Based）
先学习环境动态模型，再用模型做 planning。

代表：
- Dreamer, MuZero, PETS, MBRL

### 离线强化学习（Offline RL）
从固定数据集中学习，不允许和环境交互。

代表：CQL, IQL, Decision Transformer

### Model-based vs Model-free（机器人语境）

| 维度 | Model-free RL | Model-based RL |
|---|---|---|
| 核心思想 | 直接学策略/价值函数 | 学动力学模型 + 在模型上规划或生成数据 |
| 典型算法 | PPO, SAC, TD3 | PETS, MBPO, Dreamer |
| 样本效率 | 通常较低 | 通常更高（复用模型 rollout） |
| 偏差来源 | 高方差、探索不足 | 模型误差（model bias） |
| 工程代价 | 训练时间长，但实现路径成熟 | 系统复杂度高（模型学习 + 规划） |
| 在机器人中的常见用法 | 大规模仿真并行训练 locomotion 策略 | 在数据昂贵场景下做样本效率优化，或做混合架构 |

实践上常见折中是：
- 先用 model-based 方式快速得到可行策略
- 再用 model-free 微调，降低模型误差带来的性能上限问题

## 在机器人控制中的典型应用

- 四足/双足行走
- 人形机器人全身控制
- 机械臂操作
- 多指灵巧手操作

## 优势

- 能处理高维状态/动作空间
- 不需要精确建模
- 能发现人工难以设计的复杂策略

## 局限

- Sample efficiency 低（需要大量交互）
- Reward 设计困难
- 安全性难以保证（尤其是真实机器人上）
- 训练不稳定

## 和其他方法的关系

- **vs 模仿学习**：RL 自己探索，IL 跟随专家。IL 样本效率高但依赖专家数据；RL 可超越专家但训练难。
- **vs 最优控制**：RL model-free，最优控制 model-based。两者在 model-based RL 中逐渐融合。

## 参考来源

- Sutton & Barto, *Reinforcement Learning: An Introduction* — RL 标准教材，MDP 框架基础
- Schulman et al., *Proximal Policy Optimization Algorithms* — 机器人领域最常用的 policy gradient 算法
- [sources/papers/locomotion_rl.md](../../sources/papers/locomotion_rl.md) — locomotion RL ingest 摘要（AMP/ASE 等）
- [sources/papers/sim2real.md](../../sources/papers/sim2real.md) — sim2real 与策略迁移相关论文摘录
- [Locomotion RL 论文导航](../../references/papers/locomotion-rl.md) — 机器人 RL 应用论文集合

## 关联页面

- [Imitation Learning](./imitation-learning.md)
- [Sim2Real](../concepts/sim2real.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [Locomotion](../tasks/locomotion.md)
- [WBC vs RL](../comparisons/wbc-vs-rl.md)

## 继续深挖入口

如果你想沿着 RL 继续往下挖，建议从这里进入：

- [Robot Learning Overview](../overview/robot-learning-overview.md) — 机器人学习全景

### 论文入口
- [Locomotion RL 论文导航](../../references/papers/locomotion-rl.md)
- [Survey Papers](../../references/papers/survey-papers.md)

### 开源框架入口
- [RL Frameworks](../../references/repos/rl-frameworks.md)
- [Simulation](../../references/repos/simulation.md)
