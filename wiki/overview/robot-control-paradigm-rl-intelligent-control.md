---
type: overview
tags:
  - control
  - reinforcement-learning
  - policy-gradient
  - model-based-rl
  - hrl
status: complete
updated: 2026-07-18
summary: "RL 通过奖励驱动策略优化，涵盖值函数、策略梯度、MBRL、HRL 与模仿学习预训练。"
related:
  - ../comparisons/robot-control-eight-paradigms-taxonomy.md
  - ../methods/reinforcement-learning.md
  - ../methods/value-based-reinforcement-learning.md
  - ../methods/policy-optimization.md
  - ../methods/model-based-rl.md
  - ../methods/hierarchical-reinforcement-learning.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# 强化学习智能控制（体系⑧）

无标注环境交互试错，靠奖励函数优化策略，面向未知动力学与复杂多任务。

## 一句话定义

无标注环境交互试错，靠奖励函数优化策略，面向未知动力学与复杂多任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| MDP | Markov Decision Process | 序贯决策建模 |
| PPO | Proximal Policy Optimization | 工业常用策略梯度 |

## 为什么重要

复杂地形 loco、操作长流程任务在手工建模困难时，RL 成为 **数据驱动高层策略** 主力。

## 核心原理

MDP 上智能体选动作获奖励；值函数法估计 $Q(s,a)$；策略梯度直接优化 $\pi_\theta$；MBRL 先学模型再想象 rollout；HRL 分层子任务；BC/GAIL 用示教初始化。

## 代表性算法

| 分支 | 节点 |
|------|------|
| 值函数 | [value-based-reinforcement-learning.md](../methods/value-based-reinforcement-learning.md) |
| 策略梯度 | [policy-optimization.md](../methods/policy-optimization.md)、[ppo.md](../methods/ppo.md) |
| MBRL | [model-based-rl.md](../methods/model-based-rl.md) |
| HRL | [hierarchical-reinforcement-learning.md](../methods/hierarchical-reinforcement-learning.md) |
| 模仿学习 | [imitation-learning.md](../methods/imitation-learning.md) |

## 工程实践

Sim 大规模训练 + DR + [Sim2Real](../concepts/sim2real.md)；实机保留 PD 执行层；PPO 为人形/足式默认首选。

## 局限与风险

样本效率低、安全难、奖励设计敏感；依赖底层伺服稳定。

## 关联页面

- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Humanoid RL Cookbook](../queries/humanoid-rl-cookbook.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

