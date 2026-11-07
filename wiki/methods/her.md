---
type: method
tags: [rl, sparse-reward, manipulation, goal-conditioned, off-policy]
status: complete
updated: 2026-11-07
related:
  - ./reinforcement-learning.md
  - ./sac.md
  - ./td3.md
  - ../concepts/reward-design.md
  - ../concepts/curriculum-learning.md
  - ../comparisons/ppo-vs-sac.md
  - ../queries/reward-shaping-guide.md
original:
  - ../../sources/papers/her_reading_arxiv_1707_01495.md
sources:
  - ../../sources/papers/her_reading_arxiv_1707_01495.md
  - ../../sources/papers/policy_optimization.md
summary: "事后经验回放（HER）：对目标条件化的 off-policy RL，把失败轨迹按事后实际达到的目标重新标注并存入回放池，使稀疏二值奖励下的机械臂推、滑、抓放任务变得可学，无需塑形奖励；原论文用 DDPG + Fetch 机械臂验证。"
---

# Hindsight Experience Replay (HER)

## 一句话定义

HER 在回放池里把「没达成原目标」的轨迹，改标成「达成了它实际到达的那个目标」的成功轨迹，让稀疏二值奖励下的 off-policy 学习也能拿到有信息量的奖励信号。

在抓取或装配任务中，若机器人只有完美完成任务才得 1 分，随机探索几乎学不到东西。HER 的出发点（Andrychowicz 等，NIPS 2017，arXiv:1707.01495）是：人能从「未达到理想结果」里学到东西——射门偏右，也说明这串动作在球网更靠右时就是成功的。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HER | Hindsight Experience Replay | 把失败轨迹按事后实际达成的目标重标注后加入回放池 |
| UVFA | Universal Value Function Approximators | 让价值函数/策略同时以状态和目标为输入，HER 的目标条件化基础 |
| DDPG | Deep Deterministic Policy Gradient | 论文实验使用的连续动作 off-policy actor-critic |
| DQN | Deep Q-Network | 论文 bit-flipping 玩具实验使用的离散动作 off-policy 算法 |
| MDP | Markov Decision Process | HER 所处的序贯决策框架（见 [MDP](../formalizations/mdp.md)） |
| Reward | Reward Function | 塑造强化学习策略行为的标量反馈 |

## 为什么重要

1. **绕开奖励工程。** 机器人 RL 常需要精心设计、手调权重的塑形奖励（论文举例：叠积木用到五项需要仔细加权的代价）。HER 允许只用「是否成功」的二值信号，降低对领域知识的依赖。
2. **复杂操作首次只靠稀疏二值奖励学会。** 论文在 Fetch 机械臂上对推（pushing）、滑（sliding）、抓放（pick-and-place）三个任务，仅用二值奖励，DDPG+HER 几乎完美解决，而纯 DDPG 和带计数探索的 DDPG 基本学不会。
3. **与其他探索方法正交。** 论文指出，bit-flipping 实验的难点不在状态访问多样性，而在状态空间太大；VIME、计数探索、bootstrapped DQN 等探索增强在这里无济于事。

## 核心原理

### 多目标设定

沿用 UVFA：每个目标 $g\in G$ 对应一个谓词 $f_g:S\to\{0,1\}$，策略与价值函数输入 $(s,g)$，奖励 $r_g(s,a)=-[f_g(s)=0]$（未达成为 $-1$，达成为 $0$）。论文要求存在映射 $m:S\to G$ 使得任何状态 $s$ 都满足 $f_{m(s)}(s)=1$（例如抓放任务中 $m(s)$ 取物体位置）。

### 算法

经历一个回合 $s_0,\dots,s_T$ 之后，对每个转移 $s_t\to s_{t+1}$，除按原目标 $g$ 存入回放池外，再按策略 $S$ 采样一组额外目标 $g'$，用 $r'=r(s_t,a_t,g')$ 重新算奖励并存入。由于目标只影响动作选择，不影响环境动力学，可以任意重标注，**前提是底层算法是 off-policy 的**（论文验证了 DQN、DDPG，并提到 NAF、SDQN 同样适用）。

最简版本取 $S=m(s_T)$（回合最终状态达成的目标，称 *final*）。这样论文称至少一半被回放的轨迹含有不同于 $-1$ 的奖励，学习变简单。

## 主要技术路线：目标选取策略

| 策略 | 做法 | 论文结论 |
|------|------|----------|
| final | 用回合最后状态对应的目标 | 可解推与抓放，难解滑 |
| future | 同回合中该转移之后的 $k$ 个随机状态 | **最佳**：$k=4$ 或 $8$，也是唯一几乎完美解决 sliding 的 |
| episode | 同回合中任意 $k$ 个状态 | 次之 |
| random | 整个训练历史中随机 $k$ 个状态 | 最差 |

$k$ 过大（超过 8）会因普通回放数据占比过低而变差。论文把 HER 视作一种**隐式课程**：回放目标会从随机策略也容易达成的目标逐步转向更难的目标，但不需要控制初始状态分布。

## 工程实践

### 论文配置（Fetch，附录 A）

| 项 | 取值 |
|----|------|
| 机器人 / 仿真 | 7 自由度 Fetch 机械臂，MuJoCo；仿真训练后无微调部署实机 |
| 算法 | DDPG + Adam，学习率 $10^{-3}$，8 个线程平均参数 |
| 网络 | actor、critic 各 3 层 × 64 ReLU，actor 输出 tanh 缩放到 $[-5,5]$ cm |
| 动作 | 4 维：3 维相对夹爪位移 + 1 维指距 |
| 折扣 | $\gamma=0.98$，critic 目标裁剪到 $[-1/(1-\gamma),0]$ |
| 回放池 / batch | $10^6$ 转移；minibatch 128 |
| 训练流程 | 每 epoch 50 个 cycle，每 cycle 16 回合 + 40 次优化步；200 epoch |
| 探索 | 20% 概率均匀随机动作，否则策略输出加范围 5% 的高斯噪声 |
| 目标网络 | 每 cycle 更新一次，衰减系数 0.95 |
| 观测处理 | 归一化到均值 0、标准差 1 并裁剪到 $[-5,5]$ |
| 成功判据 | 推 / 抓放 7 cm，滑 20 cm |
| 耗时 | 8 核约 2.5 小时（推、抓放）/ 约 6 小时（滑） |

### 实践要点（均来自论文）

- **即使只关心单一目标，也训练多目标：** 论文发现多目标训练比固定单一目标学得更快。
- **塑形奖励反而更差：** 论文试验的几种塑形形式（距离的 $p$ 次方差分等）无论有无 HER 都没能解开任务，作者给出两个解释：优化目标与成功判据不一致；惩罚项抑制探索，可能学成「根本不去碰物体」。这是**该论文设置下**的观察，不代表所有任务上塑形都更差。
- **sim2real：** 抓放策略无微调部署实机，物体位置由仅用仿真渲染图 + 域随机化训练的 CNN 估计；起初 5 次成功 2 次，在观测上加 1 cm 高斯噪声重训后 5/5。
- **接触穿透：** 推任务中智能体学会把箱子压进桌面、被接触力顶出而到达目标，作者在奖励里加了穿透深度平方惩罚。
- 与本库其他 off-policy 方法的关系：HER 只是回放层面的技巧，可叠在 [SAC](./sac.md)、[TD3](./td3.md) 等 off-policy 算法之上（论文原文只验证了 DQN 与 DDPG；叠加到 SAC 属推测性迁移，需自行验证）。选型对比见 [PPO vs SAC](../comparisons/ppo-vs-sac.md)。

## 局限与风险

- **必须是 off-policy：** on-policy 算法（如 PPO）用旧数据重标注会破坏采样分布假设，HER 不能直接套用（推测性推论，论文只说明可与任意 off-policy 算法组合）。
- **需要「可映射的目标空间」：** 要能从任一状态写出对应的已达成目标 $m(s)$；目标只描述状态的一部分时依赖这一映射设计。
- **目标定义决定能学什么：** 论文的目标是物体位置，容差 $\epsilon$ 固定；对姿态、接触力等目标需要自行设计谓词。
- **论文自述的实验边界：** 任务是 Fetch 机械臂上位置目标的推 / 滑 / 抓放，无动态障碍，抓放任务最初靠「从已抓住箱子的状态起步」的技巧（后来发现若目标有时在桌面、有时在空中可不用）。
- **实机迁移不免费：** 原始策略对物体位置估计误差不鲁棒，需要观测加噪训练才稳（5 次中 2 次 → 5/5）。

## 关联页面

- [Reinforcement Learning](./reinforcement-learning.md) — HER 所属的 RL 方法总览
- [SAC](./sac.md) — 常用来与 HER 搭配的 off-policy 最大熵算法（搭配属推测）
- [TD3](./td3.md) — DDPG 的改进版，同为连续控制 off-policy 方案
- [Reward Design 概念](../concepts/reward-design.md) — 稀疏奖励与塑形奖励的权衡
- [Reward Shaping 指南](../queries/reward-shaping-guide.md) — 何时该塑形、何时用稀疏奖励 + 技巧
- [Curriculum Learning](../concepts/curriculum-learning.md) — HER 是一种隐式课程
- [PPO vs SAC](../comparisons/ppo-vs-sac.md) — on-policy 与 off-policy 选型

## 参考来源

- [HER 论文归档（原文通读）](../../sources/papers/her_reading_arxiv_1707_01495.md)
- Andrychowicz, M., et al. (2017). *Hindsight Experience Replay*. NIPS 2017. <https://arxiv.org/abs/1707.01495>
- [policy_optimization 论文档案](../../sources/papers/policy_optimization.md)

## 推荐继续阅读

- 论文原文：<https://arxiv.org/abs/1707.01495>
- Schaul et al. (2015), *Universal Value Function Approximators*（HER 的目标条件化基础，ICML 2015）
- [Spinning Up in Deep RL](https://spinningup.openai.com)
