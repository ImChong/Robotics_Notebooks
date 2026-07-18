---
type: method
tags:
  - reinforcement-learning
  - value-based
  - dqn
  - q-learning
  - discrete-control
status: complete
updated: 2026-07-18
summary: "离散动作 RL：Q-Learning → DQN → Double/Dueling 变体。"
related:
  - ../overview/robot-control-paradigm-rl-intelligent-control.md
  - ./reinforcement-learning.md
  - ./policy-optimization.md
  - ../formalizations/mdp.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
---


# Value-based Reinforcement Learning（基于值函数的强化学习）

值函数 RL：估计状态-动作价值 $Q(s,a)$ 或 $V(s)$，通过贪心或 $\epsilon$-贪心选动作，适合离散控制。

## 一句话定义

> 值函数 RL：估计状态-动作价值 $Q(s,a)$ 或 $V(s)$，通过贪心或 $\epsilon$-贪心选动作，适合离散控制。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Q-Learning | Q-Learning | 表格 Q 更新 |
| DQN | Deep Q-Network | 深度 Q 网络 |
| TD | Temporal Difference | 时序差分学习 |

## 为什么重要

简易 AGV、单关节离散任务仍可用；高维连续机器人主流已转向策略梯度。

## 核心原理

$Q(s,a)\leftarrow Q + \alpha [r + \gamma \max_{a'} Q(s',a') - Q]$；DQN 用 NN 近似 Q，经验回放与目标网络稳训练。

## 工程实践

离散化动作用于机械臂格点；Double DQN 减过估计；Dueling 分解 $V,A$。

## 主要技术路线

### 1. Q 表 / DQN 值函数

文内代表实现路径；详见 [关联概念/形式化](../formalizations/mdp.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

连续力矩控制需离散化，维数爆炸；过估计与不稳定训练。

## 关联页面

- [Reinforcement Learning](./reinforcement-learning.md)
- [Policy Optimization](./policy-optimization.md)
- [MDP](../formalizations/mdp.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>

