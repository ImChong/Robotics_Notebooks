---
type: concept
tags: [rl, reward, locomotion, humanoid, policy-optimization]
status: complete
related:
  - ../formalizations/mdp.md
  - ../methods/reinforcement-learning.md
  - ../methods/imitation-learning.md
  - ./domain-randomization.md
  - ../queries/reward-design-guide.md
summary: "Reward Design 研究如何把任务目标转成可学习的奖励信号，是机器人 RL 成败的关键工程环节。"
---

# Reward Design

**奖励函数设计（Reward Design）**：强化学习中定义智能体优化目标的核心环节。奖励函数的好坏直接决定策略能不能学出来、学出来后的行为是否符合预期。

## 一句话定义

> 奖励函数就是"你告诉机器人什么是好、什么是坏"的语言。设计得好，机器人学出你想要的；设计得差，机器人找到 reward hacking 的漏洞，表面指标满分，实际行为一塌糊涂。

## 为什么重要

MDP 框架的核心之一就是奖励函数 $R(s, a, s')$。理论上，只要奖励函数正确，RL 能找到最优策略。实践中：

- 奖励函数很难完全表达人类意图
- 策略会找到人类未预期的"钻空子"路径（reward hacking）
- 奖励项太稀疏 → 探索困难
- 奖励项太密集 → 过度约束，策略僵化

对于人形机器人 locomotion，奖励函数几乎是决定行为质量的最关键设计变量。

## 人形机器人 locomotion 中的典型奖励组成

| 奖励项 | 类型 | 作用 |
|--------|------|------|
| 线速度跟踪 | 正向 | 让机器人向目标方向走 |
| 角速度跟踪 | 正向 | 控制转向 |
| 直立惩罚 | 负向 | 防止躯干倾斜 |
| 关节力矩惩罚 | 负向 | 减少能耗和硬件磨损 |
| 接触力惩罚 | 负向 | 避免脚落地过猛 |
| 动作平滑性 | 负向 | 减少抖动和高频振荡 |
| 关节速度惩罚 | 负向 | 避免关节过快运动 |
| 摔倒终止惩罚 | 负向 | 重度惩罚摔倒行为 |
| 对称性奖励 | 正向 | 鼓励双足对称步态 |

典型加权组合形式：

$$R = \sum_i w_i r_i(s, a, s')$$

权重 $w_i$ 的调整决定了行为的优先级——是更重视速度还是稳定性，是更重视能耗还是动作幅度。

## 常见问题与反模式

### 1. Reward Hacking（奖励破解）

策略找到了满足奖励但违背真实意图的行为：

- 例子：关节速度惩罚太重 → 策略干脆不动（速度=0，奖励最高）
- 例子：高度奖励太强 → 策略保持半蹲站姿（重心低，最安全）
- 例子：速度奖励不考虑方向 → 策略倒着走也能得分

**应对方法**：人工测试边界情况；用 adversarial rollouts 检查漏洞；加入足够多的约束项。

### 2. 奖励项冲突

多个奖励项在某些状态下对动作的梯度方向相反：

- 快速行走 vs. 平稳接触
- 大步幅 vs. 关节力矩限制

**应对方法**：调整权重，优先保证硬约束（稳定、不摔），再优化软指标（速度、能耗）。

### 3. 稀疏奖励与密集奖励的权衡

- **稀疏**：只在任务成功时给奖励，探索极困难（机器人自然走到目标地点的概率极低）
- **密集**：每步都有反馈，探索容易但容易过度约束行为

**实践建议**：locomotion 一般用密集奖励，但保留少量稀疏项（如成功完成某段距离）。

## Potential-Based Reward Shaping

一种理论上不改变最优策略但改善训练效率的技术：

$$R'(s, a, s') = R(s, a, s') + \gamma \Phi(s') - \Phi(s)$$

其中 $\Phi(s)$ 是势函数（potential function），可以用于引导探索。

理论保证：只要 $\Phi$ 是状态函数，最优策略不变（Ng et al. 1999）。

## 自动化奖励设计方向

### Adversarial Motion Priors (AMP)
用判别器代替手工奖励：从参考动作数据中学一个"行为真实性"鉴别器，作为奖励信号。好处是不需要手工设计 style 奖励项，行为自然美观。

### LLM-based Reward Design
近期趋势：用 LLM 自动生成初始奖励函数，再通过迭代 RL 训练+人工反馈修正。代表：EUREKA (Ma et al. 2023)。

## 与其他页面的关系

### 和 MDP 的关系
Reward Design 是 MDP 五元组中 $R(s, a, s')$ 的工程化实现。

见：[Markov Decision Process](../formalizations/mdp.md)

### 和 RL 方法的关系
奖励函数是所有 RL 算法的优化目标，PPO/SAC 的策略梯度方向完全由奖励决定。

见：[Reinforcement Learning](../methods/reinforcement-learning.md)

### 和 Imitation Learning 的关系
IL 的核心动机之一就是"奖励函数太难设计"——让 IL 从示范数据中绕过这个问题。

见：[Imitation Learning](../methods/imitation-learning.md)

### 和 Domain Randomization 的关系
DR 改变的是环境的物理参数分布；Reward Design 改变的是优化目标。两者都影响策略，但影响的层次不同。

见：[Domain Randomization](./domain-randomization.md)

## 参考来源

- Ng et al., *Policy Invariance Under Reward Transformations* (1999) — Potential-based shaping 理论基础
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning* (2021) — locomotion reward 设计工程实践
- Ma et al., *EUREKA: Human-Level Reward Design via Coding Large Language Models* (2023) — LLM 自动奖励设计
- **ingest 档案：** [sources/papers/policy_optimization.md](../../sources/papers/policy_optimization.md) — PPO/SAC locomotion reward 实践
- **ingest 档案：** [sources/papers/privileged_training.md](../../sources/papers/privileged_training.md) — Walk These Ways 步态条件化 reward

## 推荐继续阅读

- [Markov Decision Process](../formalizations/mdp.md)
- [Policy Optimization](../methods/policy-optimization.md)
- [Imitation Learning](../methods/imitation-learning.md)（AMP 路线：绕过手工奖励）
- [Query：Reward Design 实战指南](../queries/reward-design-guide.md)

## 一句话记忆

> 奖励函数是"你告诉机器人什么算成功"的唯一语言——设计得好，RL 帮你把行为做出来；设计得差，RL 帮你把漏洞钻透。
