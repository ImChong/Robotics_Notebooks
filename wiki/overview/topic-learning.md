---
type: overview
tags: [topic, topic-learning, rl, il, imitation, reinforcement]
status: complete
updated: 2026-06-17
summary: "IL/RL 学习范式专题汇总：强化学习、模仿学习、行为克隆与 model-based 路线的选型、数据需求与机器人落地注意点。"
---

# IL/RL 学习范式（专题汇总）

> **图谱专题视图**：本页是知识图谱「🎓 模仿/强化学习 (IL/RL)」专题的统一入口；在 [图谱专题视图](../../docs/graph.html?topic=learning) 筛选时，本节点为汇总锚点。

## 一句话定义

**机器人学习专题** 覆盖 **从数据或交互中习得策略** 的主要范式：强化学习（RL）、模仿学习（IL）及其与 model-based、离线 RL、基础模型的组合。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 从奖励信号学习策略 |
| IL | Imitation Learning | 从专家示范学习 |
| BC | Behavior Cloning | 监督式模仿 |
| PPO | Proximal Policy Optimization | 常用 on-policy RL 算法 |
| SAC | Soft Actor-Critic | 常用 off-policy 连续控制算法 |

## 为什么重要

- **现代机器人技能主路径**：Locomotion、WBT、操作大量依赖 RL/IL。
- **范式选型决定数据与工程成本**：有示范优先 IL，需探索用 RL，有模型走 MBRL。
- **与 Sim2Real / 安全微调强耦合**：学习只在仿真完成一半，落地还有后半段。

## 本专题覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 总览 | 机器人学习全景 | [Robot Learning Overview](./robot-learning-overview.md) |
| 方法 | RL / IL | [Reinforcement Learning](../methods/reinforcement-learning.md)、[Imitation Learning](../methods/imitation-learning.md) |
| 对比 | RL vs IL / PPO vs SAC | [RL vs IL](../comparisons/rl-vs-il.md)、[PPO vs SAC](../comparisons/ppo-vs-sac.md) |
| 概念 | 奖励设计 / 课程 | [Reward Design](../concepts/reward-design.md)、[Curriculum Learning](../concepts/curriculum-learning.md) |
| 概念 | 特权训练 / 想象 | [Privileged Training](../concepts/privileged-training.md) |

## 与其他专题的关系

- **[Sim2Real](./topic-sim2real.md)**：学习策略如何上真机。
- **[WBT](./topic-wbt.md)**：跟踪与 AMP 大量用 RL。
- **[安全微调](./topic-safe-fine-tuning.md)**：部署后在线 RL 的安全边界。

## 关联页面

- [Model-Based vs Model-Free](../comparisons/model-based-vs-model-free.md)
- [Online vs Offline RL](../comparisons/online-vs-offline-rl.md)
- [Deep RL Game Milestones](../concepts/deep-rl-game-milestones.md)

## 参考来源

- 本库归纳自 [Robot Learning Overview](./robot-learning-overview.md) 及 methods/comparisons 学习系列页
- 图谱专题定义：[docs/topic-filters.js](../../docs/topic-filters.js)（`learning` 命中规则）
