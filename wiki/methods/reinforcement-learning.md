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

## 关联页面

- [Imitation Learning](./imitation-learning.md)
- [Sim2Real](../concepts/sim2real.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [Locomotion](../tasks/locomotion.md)
- [WBC vs RL](../comparisons/wbc-vs-rl.md)
