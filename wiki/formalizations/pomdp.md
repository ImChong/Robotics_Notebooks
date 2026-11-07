---
type: formalization
tags: [rl, math, decision-making, perception, uncertainty]
status: complete
updated: 2026-11-07
related:
  - ./mdp.md
  - ../concepts/state-estimation.md
  - ../concepts/bayesian-belief-analysis.md
  - ../concepts/embodied-rl-minimal-closed-loop.md
  - ./bellman-equation.md
original:
  - ../../sources/papers/kaelbling_littman_cassandra_1998_pomdp.md
  - ../../sources/books/sutton_barto_rl_intro_2nd_ed.md
sources:
  - ../../sources/papers/kaelbling_littman_cassandra_1998_pomdp.md
  - ../../sources/books/sutton_barto_rl_intro_2nd_ed.md
  - ../../sources/papers/sac_reading_arxiv_1801_01290.md
  - ../../sources/papers/bayesian_analysis_rl_primary_refs.md
  - ../../sources/papers/rl_foundation_models.md
summary: "部分可观测马尔可夫决策过程（POMDP）在 MDP 上增加观测空间与观测函数：智能体看不到状态，只能靠历史或 belief（状态的概率分布）决策；belief 是历史的充分统计量，POMDP 等价于连续空间的 belief MDP，是状态估计、记忆策略与鲁棒控制的理论基础。"
---

# Partially Observable MDP (POMDP)

## 一句话定义

POMDP 是「智能体看不到真实状态，只能看到带噪观测」的 [MDP](./mdp.md)：最优决策依赖对状态的**信念（belief，状态的概率分布）**，而不是最近一次观测。

真实机器人几乎总是这种情形：传感器噪声、视觉遮挡和未知的物理参数使我们永远拿不到完美的状态 $s$，这就是**部分可观测（partial observability）**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| POMDP | Partially Observable Markov Decision Process | 部分可观测的 MDP，部署时观测受限的常见建模 |
| MDP | Markov Decision Process | 状态–动作–奖励–转移的标准序贯决策建模框架 |
| SE | State Estimator | Kaelbling 等人的结构图里负责更新 belief 的模块 |
| EKF | Extended Kalman Filter | 在连续空间维护高斯 belief 的常用状态估计器 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |

## 为什么重要

1. **它是真机部署的默认假设。** 仿真里策略常能读到特权状态；部署时只有 IMU、编码器、噪声视觉。把问题误当 MDP，会让只看当前观测的策略在「外观相同、状态不同」的情形下反复做同一件事。Kaelbling 等人在原文中就点明：把最近观测直接映射成动作、不记忆历史，对看起来一样的不同位置会做同样的事；确定性观测–动作映射还容易陷入循环。
2. **它统一了「改变世界」与「获取信息」。** 在 POMDP 里没有专门区分这两类动作，信息价值通过观测函数自然体现，机器人为了看清而移动（主动感知）与为了到达目标而移动是同一个优化问题的两面。
3. **它解释了为什么需要记忆。** Sutton & Barto 教材 §17.3 指出：状态可以看作历史的紧凑、可递归更新的摘要；POMDP 的 belief 就是这样一种马尔可夫状态。后续「历史堆叠 / RNN / Transformer 编码隐状态」都可以看作对它的近似。

## 核心原理

### 形式定义

Kaelbling、Littman 与 Cassandra（1998）把 POMDP 写成六元组 $\langle S, A, T, R, \Omega, O\rangle$：

- $S, A, T, R$ 与 [MDP](./mdp.md) 相同：$T(s,a,s')$ 为转移概率，$R(s,a)$ 为期望即时奖励；
- $\Omega$：有限观测集；
- $O(s', a, o)$：执行动作 $a$ 并落入状态 $s'$ 后观测到 $o$ 的概率。

原文脚注说明，观测也可以建模为依赖前一状态，两者表达力等价，只是需扩充状态空间。目标仍是最大化期望折扣回报（原文取 $0<\gamma<1$）。

### Belief 与状态估计器

智能体拆成两部分（原文图 2）：状态估计器 SE 根据「上一个 belief $b$、刚执行的动作 $a$、新观测 $o$」产出新 belief $b'$；策略 $\pi$ 以 belief 为输入输出动作。

$$ b'(s') = \frac{O(s',a,o)\sum_{s\in S} T(s,a,s')\,b(s)}{\Pr(o\mid a,b)} $$

分母 $\Pr(o|a,b)$ 只是归一化因子，使 $b'$ 之和为 1。原文强调 belief 必须是完整分布而不是「最可能的状态」，因为智能体要把自己的不确定性考虑进去（迷路时应当去查地图或找路标）；并且 belief 是历史与初始 belief 的**充分统计量**，所以 belief 上的过程是马尔可夫的。

> 与本页旧版的差别：旧版把 belief 写成 $b_t(s)=P(s_t=s\mid o_{1:t},a_{1:t-1})$，这是等价的另一种写法；上式给出的是递推形式，也是实现时真正用到的形式。

### Belief MDP

把 belief 当作状态，POMDP 变成连续状态空间的 MDP：状态集 $B$、转移 $\tau(b,a,b')=\sum_o \Pr(b'\mid b,a,o)\Pr(o\mid a,b)$、奖励 $\rho(b,a)=\sum_s b(s)R(s,a)$。该 belief MDP 的最优策略配合正确的状态估计器，给出原 POMDP 的最优行为。

### 价值函数的结构

用深度为 $t$ 的**策略树**（节点是动作，边是观测）表示 $t$ 步非平稳策略，每棵树对应一个 $|S|$ 维向量 $\alpha_p$，则 $V_p(b)=b\cdot\alpha_p$，最优 $t$ 步价值函数 $V_t(b)=\max_p b\cdot\alpha_p$。因此有限步最优价值函数是 belief 空间上的**分段线性凸函数**；无限时域折扣情形仍为凸，但可能有无穷多「面」，可任意逼近。

### 教材视角（Sutton & Barto §17.3）

教材不预设潜在状态：环境只给观测，交互写成 $A_0,O_1,A_1,O_2,\dots$；历史 $H_t=A_0,O_1,\dots,A_{t-1},O_t$ 是最完整的信息；「状态」是历史的函数 $S_t=f(H_t)$，并要求能递归更新 $S_{t+1}=u(S_t,A_t,O_{t+1})$。若 $f$ 能无损预测一切未来，就是**马尔可夫状态**。POMDP 是取得马尔可夫状态的一种做法：假设有一个对智能体不可见的潜在状态，智能体维护它的分布。

## 工程实践

1. **状态估计：** [EKF](../concepts/state-estimation.md) 本质上是在连续空间维护高斯 belief 的 SE。注意：Kaelbling 等人在原文引言中专门指出，卡尔曼类方法用椭球 / 正态分布表示不确定性，对「离散不确定性」（几乎确定在某两层楼之一）并不合适，所以 belief 取一般概率分布。
2. **主动感知（Active Perception）：** 机器人通过移动传感器减小 $b_t$ 的不确定度，其价值体现在后续奖励里。
3. **深度 RL 里的近似做法：** 把最近若干步的观测与动作拼成输入（SAC 续作 arXiv:1812.05905 的 Minitaur 真机实验就用当前和过去 5 步观测与动作，原因是延迟与接触使动力学非马尔可夫，见 [SAC 论文归档](../../sources/papers/sac_reading_arxiv_1801_01290.md)），或用 RNN / Transformer 编码历史；特权信息教师–学生蒸馏（见 [Teacher-Student DAgger](../methods/teacher-student-dagger-training.md)）则是让学生从可观测历史里重建教师看到的特权量。这些在 Kaelbling 原文之外，属于本页对工程做法的概括（推测性映射，非原文内容）。
4. **最小闭环：** 见 [具身 RL 最小闭环](../concepts/embodied-rl-minimal-closed-loop.md) 对 MDP 与 POMDP 的衔接。

## 局限与风险

- **精确求解只适用于小问题：** 原文承认即使 witness 算法，在 $|S|>15$ 且 $|\Omega|>15$ 的中等规模问题上就变得不实用；单次价值迭代所需策略树数量可以指数级增大。
- **需要已知完整正确的模型：** 精确 belief 更新要 $T$ 与 $O$；原文把模型学习列为未涉及的方向。模型错了，belief 会系统性偏离。
- **要求状态可枚举：** 原文方法需要有限状态的枚举表示，而非组合式表示。
- **一步预测误差会累积：** 教材提醒，若用一步预测迭代出长期预测，一步预测的任何误差都会被放大，实践中经常如此。
- **不要把「函数逼近隐含的部分可观测」当成解决了问题：** 教材指出参数化函数逼近可以让价值函数不依赖某些不可观测变量，等效于它们不可观测，但这并不能自动给出记忆或 belief。

## 关联页面

- [MDP 形式化](./mdp.md) — POMDP 退化到完全可观测时的特例
- [Bayesian Belief Analysis（贝叶斯信念分析）](../concepts/bayesian-belief-analysis.md) — belief 更新、Bayes 滤波与 BRL 总览
- [State Estimation (状态估计)](../concepts/state-estimation.md) — EKF 等高斯 belief 估计器
- [Bellman 方程](./bellman-equation.md) — belief MDP 上的价值递推同样遵循 Bellman 结构
- [具身 RL 最小闭环](../concepts/embodied-rl-minimal-closed-loop.md) — 真机部署为何几乎总是 POMDP
- [Teacher-Student DAgger 训练](../methods/teacher-student-dagger-training.md) — 特权教师到可观测学生的蒸馏

## 参考来源

- [Kaelbling, Littman, Cassandra (1998) 论文归档](../../sources/papers/kaelbling_littman_cassandra_1998_pomdp.md) — 通读：POMDP 定义、belief 更新、belief MDP、分段线性凸、witness 算法
- [Sutton & Barto 教材归档](../../sources/books/sutton_barto_rl_intro_2nd_ed.md) — §17.3 观测与状态
- [贝叶斯分析一手资料索引](../../sources/papers/bayesian_analysis_rl_primary_refs.md)
- Kaelbling, L. P., Littman, M. L., Cassandra, A. R. (1998). *Planning and Acting in Partially Observable Stochastic Domains*. Artificial Intelligence 101:99–134.

## 推荐继续阅读

- [Kaelbling 等人论文 PDF（作者主页）](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf)
- [Sutton & Barto 教材（第 17.3 节）](http://incompleteideas.net/book/RLbook2020.pdf)
