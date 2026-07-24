---
type: method
tags:
  - reinforcement-learning
  - hrl
  - hierarchical
  - long-horizon
status: complete
updated: 2026-07-24
summary: "移动-抓取-放置等长流程任务的层次化 RL。"
related:
  - ../overview/robot-control-paradigm-rl-intelligent-control.md
  - ./reinforcement-learning.md
  - ../concepts/curriculum-learning.md
  - ../entities/paper-aware-wheeled-legged-reflexive-evasion.md
sources:
  - ../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md
  - ../../sources/papers/aware_arxiv_2604_23761.md
---


# Hierarchical Reinforcement Learning（分层强化学习，HRL）

HRL：上层策略拆分子任务/选项，下层策略执行具体运动，缓解长时程信用分配与探索难题。

## 一句话定义

> HRL：上层策略拆分子任务/选项，下层策略执行具体运动，缓解长时程信用分配与探索难题。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HRL | Hierarchical Reinforcement Learning | 分层强化学习 |
| Option | Temporal Option | 时间抽象子策略 |
| Manager | Manager Policy | 上层任务分配 |

## 为什么重要

复杂操作任务纯扁平 RL 难收敛，HRL 提供 **时间抽象**。

## 核心原理

上层 $\pi_{high}(o|s)$ 选 option $o$，下层 $\pi_{low}(a|s,o)$ 执行；Option-Critic 端到端训选项。

## 工程实践

人形 loco-manip 分「行走/操作」层；分层 PPO；子目标奖励塑形。

**轮足高动态避障实例：** [AWARE](../entities/paper-aware-wheeled-legged-reflexive-evasion.md) 用高层输出 \(v_{\mathrm{cmd}}\) + Gumbel-Softmax 模态，再 **硬路由** 到低速全向 / 高动态敏捷两个预训练专家（非 MoE 软混合），适合「导航避让 ↔ 反射逃逸」两套动力学并存的场景。

## 主要技术路线

### 1. 上层选项 + 下层子策略

文内代表实现路径；详见 [关联概念/形式化](../concepts/curriculum-learning.md)。

### 2. 与相邻体系融合

常与 [八大机器人控制体系分类](../comparisons/robot-control-eight-paradigms-taxonomy.md) 中相邻类别 **级联或并联**（如 CTC+SMC、MPC+ILC、NN 补偿+PID）。

## 局限与风险

层间接口设计难；错误子目标导致下层无法完成。

## 关联页面

- [Reinforcement Learning](./reinforcement-learning.md)
- [Curriculum Learning](../concepts/curriculum-learning.md)
- [AWARE（轮足分层反射避障）](../entities/paper-aware-wheeled-legged-reflexive-evasion.md)

## 参考来源

- [wechat_shenlan_robot_control_eight_paradigms.md](../../sources/blogs/wechat_shenlan_robot_control_eight_paradigms.md) — 深蓝具身智能《机器人控制算法八大体系详解：从 PID 到强化学习》（<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>）
- [AWARE（arXiv:2604.23761）](../../sources/papers/aware_arxiv_2604_23761.md) — 轮足双专家硬切换分层实例

## 推荐继续阅读

- 深蓝具身智能原文：<https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g>
- [AWARE 论文](https://arxiv.org/abs/2604.23761) — 速度指令层 + 离散专家路由

