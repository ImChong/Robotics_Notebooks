---
type: method
tags: [il, dagger, online-learning, covariate-shift, expert-intervention]
updated: 2026-06-07
status: complete
summary: "DAgger 通过让当前策略访问状态、再由专家回标这些状态，系统性缓解 Behavior Cloning 的分布漂移问题。"
related:
  - ./behavior-cloning.md
  - ./imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/ultra-survey.md
  - ../comparisons/rl-vs-il.md
sources:
  - ../../sources/papers/imitation_learning.md
---

# DAgger（Dataset Aggregation）

**DAgger**：一种交互式模仿学习方法，让当前策略先去“自己跑”，再由专家为这些真实访问到的状态打标签，并把新数据持续并入训练集。

## 一句话定义

DAgger 的核心不是换一个模型，而是换一个数据收集闭环：让策略暴露自己的错误分布，再让专家补这些坑。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DAgger | Dataset Aggregation | 迭代收集策略诱导状态下的专家标注以纠偏的模仿学习方法 |
| BC | Behavior Cloning | 将状态映射到动作的监督式模仿，易受分布偏移影响 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| IL | Imitation Learning | 从专家演示学习策略，奖励难定义时的主路线 |
| BFM | Behavior Foundation Model | 大规模行为数据预训练的可复用全身行为先验 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |

## 为什么重要

Behavior Cloning 的根本问题不是监督学习本身，而是**训练分布和部署分布不一致**。DAgger 用在线数据聚合把“策略会去到哪里”纳入训练集，因此通常比纯 BC 更数据有效，也更适合长时序闭环控制。

## 主要技术路线

设专家策略为 $\pi^*$，学习策略为 $\pi_	heta$：

1. 用少量专家轨迹初始化训练集 $D_0$
2. 训练初始策略 $\pi_1$
3. 用 $\pi_i$ 在环境中 rollout，收集其访问到的状态 $s$
4. 让专家为这些状态标注动作 $\pi^*(s)$
5. 将新样本并入数据集：$D_i = D_{i-1} \cup \{(s, \pi^*(s))\}$
6. 在聚合数据集上重新训练或增量更新策略

## DAgger 为什么比 BC 更有效

### 1. 直接覆盖部署分布
策略一旦发生轻微偏移，就会进入专家演示中罕见的状态。DAgger 专门把这些状态收集回来，因此训练分布逐步逼近部署分布。

### 2. 减少 compounding error
它不能让错误消失，但能让模型见过“出错后的恢复动作”，这正是长 horizon 任务最需要的数据。

### 3. 更适合安全干预式数据收集
在真机上常见实现不是完全让策略失控，而是**共享控制 / 专家接管 / 人类纠偏**。这比离线 BC 更接近真实部署流程。

## 和 BC 的对比

| 维度 | Behavior Cloning | DAgger |
|------|------------------|--------|
| 数据来源 | 纯专家离线演示 | 当前策略 rollout + 专家回标 |
| 是否处理 covariate shift | 弱 | 强 |
| 长时序鲁棒性 | 易累积误差 | 通常更好 |
| 标注成本 | 低 | 更高，需要在线专家参与 |
| 真机风险 | 低 | 需要安全回退机制 |

## 工程落地点

### 操作任务
- 用遥操作或共享控制做回标
- 收集“快要失败但尚未完全失败”的纠偏数据
- 比纯 BC 更适合插拔、接触、双手协调等长时序任务

### locomotion / whole-body 任务
- 可以把高性能教师控制器、MPC 或人类设计的参考轨迹当专家
- 更常见的变体是 teacher-student 蒸馏，而非完全照搬原始 DAgger 形式
- **[PHP](../entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md)**：高动态跑酷学生策略用 **DAgger + PPO** 混合损失；纯 DAgger 对攀爬/翻越不足，需 success-driven RL 项
- **[RPL](../entities/paper-rpl-robust-humanoid-perceptive-locomotion.md)**：分地形高程 **专家** 以 **DAgger 动作回归** 蒸馏为 **多视角深度** 统一下身策略；辅以 DFSV/RSM 处理多向与非对称感知
- **[AssistMimic](../entities/paper-assistmimic.md)**：双人 assistive tracking 的 **generalist** 用 DAgger 蒸馏多 subject specialist（arXiv:2603.11346）

## 潜在坑

- **专家成本**：真机标注最贵的不是训练，而是在线专家参与。
- **安全性**：策略 rollout 时必须有硬保护，不然为了覆盖错误状态可能直接摔机。
- **数据偏置**：如果专家总是过早接管，数据集中会缺少“接近失败但可恢复”的状态。
- **训练不稳定**：每轮数据分布都在变，最好有明确的 replay / weighting 策略。

## 常见误区

- **误区 1：DAgger 与 BC 没本质区别。**
  错。它们的训练损失可以相似，但数据分布完全不同。
- **误区 2：DAgger 不处理 covariate shift。**
  错。它最核心的目标就是处理这一点。
- **误区 3：DAgger 一定更省标注。**
  不一定。它往往更省“为达到同样鲁棒性所需的无效离线演示”，但在线专家时间仍昂贵。

## 参考来源

- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md) — Ross et al. DAgger 原论文与 IL 路线摘要
- Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* — DAgger 原论文
- **ingest 档案：** [sources/papers/bfm_humanoid_arxiv_2509_13780.md](../../sources/papers/bfm_humanoid_arxiv_2509_13780.md) — BFM：DAgger 风格的掩码在线蒸馏，把人形多控制接口统一进 CVAE 学生策略
- **ingest 档案：** [sources/papers/php_parkour_arxiv_2602_15827.md](../../sources/papers/php_parkour_arxiv_2602_15827.md) — PHP：teacher-student 跑酷中 DAgger+PPO 课程蒸馏
- **ingest 档案：** [sources/papers/rpl_arxiv_2602_03002.md](../../sources/papers/rpl_arxiv_2602_03002.md) — RPL：分地形高程专家 → 多视角深度 DAgger 蒸馏

## 关联页面

- [Behavior Cloning](./behavior-cloning.md)
- [Behavior Cloning Loss](../formalizations/behavior-cloning-loss.md) — DAgger 在线聚合后依然优化的底层目标函数
- [Imitation Learning](./imitation-learning.md)
- [Manipulation](../tasks/manipulation.md)
- [ULTRA：统一多模态 loco-manipulation 控制](../tasks/ultra-survey.md)
- [RL vs Imitation Learning](../comparisons/rl-vs-il.md)

## 推荐继续阅读

- Ross et al., *DAgger* 原论文
- Chi et al., *Diffusion Policy* — 看生成式 IL 如何与交互式数据结合
- Open X / RT 系列博客与实现 — 了解大规模演示下如何做 teacher correction
