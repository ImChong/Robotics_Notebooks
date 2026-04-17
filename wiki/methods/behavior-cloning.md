---
type: method
tags: [il, behavior-cloning, supervised-learning, manipulation, covariate-shift]
status: complete
summary: "Behavior Cloning 把专家演示转成监督学习问题，是机器人模仿学习最简单也最常用的基线。"
related:
  - ./imitation-learning.md
  - ./dagger.md
  - ./diffusion-policy.md
  - ../comparisons/rl-vs-il.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/imitation_learning.md
  - ../../sources/papers/diffusion_and_gen.md
---

# Behavior Cloning（行为克隆）

**Behavior Cloning, BC**：把专家演示数据当作监督学习数据集，直接学习从观测到动作的映射，是模仿学习最直接的做法。

## 一句话定义

给机器人一堆“专家在这个状态下应该怎么做”的样本，训练一个策略去直接复现这些动作。

## 为什么重要

- 它几乎是所有模仿学习 pipeline 的起点：先训一个能跑的 BC baseline，再谈 DAgger、Diffusion Policy 或 IL+RL。
- 在奖励函数很难设计、但演示数据容易拿到的任务里，BC 往往是最低门槛方案。
- 许多真机操作系统都会先用 BC 做 warm start，再用更复杂方法提升鲁棒性。

## 输入、输出与训练目标

给定专家数据集 $D = \{(o_i, a_i)\}$，BC 通常优化：

$$
\min_	heta \; \mathbb{E}_{(o,a)\sim D}[\ell(\pi_	heta(o), a)]
$$

常见设定：
- **输入**：图像、关节状态、末端位姿、历史动作等观测
- **输出**：关节目标、末端动作、action chunk 或离散动作 token
- **损失**：MSE、L1、交叉熵、负对数似然

## 核心优点

### 1. 简单直接
它不需要环境交互、不需要在线探索，也不需要 reward engineering。

### 2. 数据效率高于纯 RL
在固定专家数据上训练，通常比从零探索的 RL 更快进入“能做事”的区间。

### 3. 工程上容易落地
训练和部署都像标准 supervised learning，适合先做 baseline、集成到已有感知模型、或作为大模型动作头。

## 核心局限

### 1. Covariate Shift / Distribution Shift
训练时看到的是专家访问到的状态，部署时策略一旦出错，就会进入训练集中没见过的状态分布。

### 2. Compounding Error
单步小误差会沿着闭环执行不断累积，序列越长、任务越长 horizon，问题越明显。BC 并不是“每步都独立无害”的方法。

### 3. 受限于专家上界
如果数据里没有恢复动作、异常姿态或罕见接触，BC 通常也学不会这些行为。

## 典型缓解策略

| 问题 | 常见缓解 |
|------|---------|
| 分布漂移 | DAgger 在线聚合策略访问到的新状态 |
| 长时序误差 | Action Chunking、序列模型、闭环再规划 |
| 多模态动作 | Diffusion Policy、Flow Matching、Mixture Density |
| 真机鲁棒性不足 | 数据增强、传感器噪声注入、真实数据微调 |

## 与 DAgger、Diffusion Policy 的关系

- **BC**：最简单，离线监督学习基线。
- **DAgger**：仍然学监督映射，但会反复收集“当前策略真正会访问到的状态”，核心是修复 covariate shift。
- **Diffusion Policy / π₀**：仍可看作 BC 范式的生成式升级，重点解决多模态动作和长时序建模。

## 在机器人里的典型应用

### 操作
- 桌面抓取、装配、双手协作
- 遥操作数据蒸馏为离线策略
- VLA / Foundation Policy 的动作头微调基线

### 移动与 locomotion
- 用 MoCap、教师策略或 privileged teacher 生成数据，再做学生策略蒸馏
- 常作为 IL+RL 混合流程的第一步，而不是最终控制器

## 常见误区

- **误区 1：BC 与 DAgger 等价。**
  不是。DAgger 的关键价值正是在于持续覆盖策略部署分布，通常比纯 BC 更能处理分布漂移。
- **误区 2：BC 的累积误差和序列长度无关。**
  错。horizon 越长，早期偏差越容易滚雪球。
- **误区 3：只要模型够大，BC 就天然鲁棒。**
  模型容量能帮助拟合，但不能替代分布覆盖。

## 参考来源

- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md) — DAgger / BC / ACT / Diffusion Policy 的 ingest 档案
- [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — 生成式模仿学习如何扩展传统 BC
- Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* — 解释为什么纯 BC 会受到 covariate shift 影响

## 关联页面

- [Imitation Learning](./imitation-learning.md)
- [DAgger](./dagger.md)
- [Diffusion Policy](./diffusion-policy.md)
- [Manipulation](../tasks/manipulation.md)
- [RL vs Imitation Learning](../comparisons/rl-vs-il.md)

## 推荐继续阅读

- Ross et al., *DAgger* — 经典交互式 IL 方法
- Zhao et al., *ACT* — 用 action chunking 缓解长时序误差
- Chi et al., *Diffusion Policy* — 生成式方法如何超越传统 BC
