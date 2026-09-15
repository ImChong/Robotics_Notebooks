# 具身智能 | 深度强化学习的两条主线：Off-policy 与 On-policy 的完整演进

> 来源归档（blog / 微信公众号）

- **标题：** 具身智能 | 深度强化学习的两条主线：Off-policy 与 On-policy 的完整演进
- **类型：** blog
- **作者：** PinkRobot（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/gfL23ksN1FeXNTp5kpUpsw
- **发表日期：** 2026-09-15（入库日）
- **入库日期：** 2026-09-15
- **抓取方式：** WebFetch（`mp.weixin.qq.com`）
- **一句话说明：** 万字教学文：以 **行为策略 π_b vs 目标策略 π** 为严格分界（非「有没有 Replay Buffer」），梳理 Off-policy（Q-learning→DQN→DDPG→TD3/SAC）与 On-policy（REINFORCE→A2C→TRPO→PPO）两条演进链；强调 Actor-Critic 为共享结构、SAC≠TD3+熵、PPO 多 epoch 仍属 on-policy，以及机器人场景选型（真机贵→off-policy / 大规模仿真→PPO）。

## 核心摘录（归纳，非全文）

### 严格定义（文内 §0）

| 概念 | 定义 |
|------|------|
| **On-policy** | 行为策略 = 目标策略（$\pi_b = \pi$） |
| **Off-policy** | 允许 $\pi_b \neq \pi$；Replay Buffer 是常见工程手段，**不是定义** |
| **Actor-Critic** | 结构分工（Actor 选动作 / Critic 评价值）；**DDPG/TD3/SAC 也是 AC**，并非 on-policy 专属 |

### 两条演进链（文内 §2、§12）

**Off-policy：** Q-learning → DQN → DDPG → TD3 / SAC（SAC 与 TD3 **同期**，根基是最大熵 RL + 随机 Actor，非「TD3+熵」）

**On-policy：** REINFORCE → Baseline/Actor-Critic → GAE → A3C/A2C → TRPO → PPO-Clip

### 各算法「解决什么问题」（文内 §12 末）

| 算法 | 核心问题 |
|------|----------|
| Q-learning | Off-policy TD control |
| DQN | 深度网络 + Q-learning |
| DDPG | 连续动作无法 argmax |
| TD3 | Actor 利用 Critic 过估计 |
| SAC | Off-policy 连续控制中的探索与稳定性 |
| REINFORCE | 直接优化策略 |
| Actor-Critic | 降低 PG 方差 |
| GAE | Advantage bias–variance 权衡 |
| TRPO | 限制单次策略更新幅度 |
| PPO | 一阶近似 trust-region |

### 机器人选型（文内 §20–22、§54）

- **真机/贵交互：** Replay 复用有价值 → TD3/SAC 等 off-policy
- **大规模 GPU 并行仿真（如 4096 env）：** 采样便宜 → PPO 结构简单、易向量化、distribution mismatch 小
- **现代 PPO 运控栈常组合：** DR、Privileged/Asymmetric AC、Curriculum、AMP、Motion Tracking、Teacher–Student、对称增广等

### 常见误区（文内强调）

1. 「有 Buffer = off-policy」——不严格
2. 「SAC = TD3 + entropy」——理论出发点不同
3. 「PPO 一批数据训多 epoch = off-policy」——仍丢弃 rollout、用当前 π 采样，属 near-on-policy
4. 「PPO 超参不敏感」——相对 TRPO 成立，但 lr/clip/GAE λ/reward scale 仍影响大

## 对 wiki 的映射

- **新建：** [deep-rl-off-on-policy-evolution](../../wiki/overview/deep-rl-off-on-policy-evolution.md)
- **交叉补强：** [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[policy-optimization](../../wiki/methods/policy-optimization.md)、[ppo-vs-sac](../../wiki/comparisons/ppo-vs-sac.md)、[PPO](../../wiki/methods/ppo.md)、[SAC](../../wiki/methods/sac.md)
