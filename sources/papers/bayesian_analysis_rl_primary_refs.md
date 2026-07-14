# 贝叶斯分析（Belief / Bayesian RL）一手资料索引

> 来源归档（ingest）

- **标题：** 贝叶斯信念推理、POMDP 与 Bayesian Reinforcement Learning 经典论文与教材
- **类型：** paper / textbook / survey（合集）
- **入库日期：** 2026-07-14
- **一句话说明：** 汇总 belief state、Bayes 滤波、POMDP 最优控制与 Bayesian RL 一手来源，作为 `wiki/concepts/bayesian-belief-analysis.md` 与 [pomdp](../../wiki/formalizations/pomdp.md) 的原始依据。
- **沉淀到 wiki：** 是 → [bayesian-belief-analysis](../../wiki/concepts/bayesian-belief-analysis.md)、[pomdp](../../wiki/formalizations/pomdp.md)、[state-estimation](../../wiki/concepts/state-estimation.md)

## 为什么值得保留

- **贝叶斯分析** 在机器人与 RL 中指：用 **概率分布（belief）** 表示对隐状态的不确定性，并按 **Bayes 规则** 递推更新——POMDP 的 belief-MDP、EKF 的高斯 belief、Bayesian RL 对模型/价值先验均属此族。
- [Richard Sutton 的 *One-Step Trap*](../blogs/sutton_one_step_trap.md) 把 **POMDP / Bayesian 分析** 与一步模型 rollout 并列为常见误用场景：维护完整 belief 再向前展开，在随机策略下分支 **指数增长**。理解一手文献才能分清「何时贝叶斯推理必要」与「何时应改用时序抽象 / GVF」。
- 机器人 **状态估计**（[Kalman Filter](../../wiki/formalizations/kalman-filter.md)）、**部分可观测 RL**（RNN 策略、非对称 actor-critic）与 **探索–利用**（Bayes-adaptive MDP）共享同一数学底座。

## 核心摘录

### 1) Smallwood & Sondik (1973) — POMDP 信念 MDP 与最优性结构

- **来源：** R. D. Smallwood, E. J. Sondik, *The Optimal Control of Partially Observable Markov Processes over a Finite Horizon*, Operations Research, 21(5):1071–1088, 1973.
- **要点：**
  - 证明 POMDP 最优价值函数在 **belief 单纯形** 上为 **分段线性凸（PWLC）** 函数；策略可写为 belief → action 映射。
  - 建立 **belief state** $b_t(s)=P(s_t=s\mid o_{1:t},a_{1:t-1})$ 作为 **充分统计量** 的理论基础——后续 POMDP 求解与近似均由此出发。
- **对 wiki 的映射：** [bayesian-belief-analysis](../../wiki/concepts/bayesian-belief-analysis.md)、[pomdp](../../wiki/formalizations/pomdp.md)

### 2) Åström (1965) — 最优控制与部分可观测（早期 POMDP 思想）

- **来源：** K. J. Åström, *Optimal Control of Markov Processes with Incomplete State Information*, Journal of Mathematical Analysis and Applications, 10:174–205, 1965.
- **要点：**
  - 在 **不完全状态信息** 下把控制问题化为对 **后验分布** 的决策；与分离原理、LQG 一脉相承。
  - 为「用概率状态做最优控制」提供连续时间/离散时间早期范本。
- **对 wiki 的映射：** [bayesian-belief-analysis](../../wiki/concepts/bayesian-belief-analysis.md)、[kalman-filter](../../wiki/formalizations/kalman-filter.md)

### 3) Kaelbling, Littman & Cassandra (1998) — POMDP 规划与学习综述

- **来源：** L. P. Kaelbling, M. L. Littman, A. R. Cassandra, *Planning and Acting in Partially Observable Stochastic Domains*, Artificial Intelligence, 101(1–2):99–134, 1998. [作者页](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf)
- **要点：**
  - 系统化 **belief 更新**、**有限 horizon 动态规划** 与 **point-based 近似**；给出 POMDP 在 AI/机器人中的标准问题表述。
  - 明确 **belief-MDP** 直觉：把 belief 当作「宏状态」，在 belief 空间上做 MDP 规划——但 belief 维数随隐状态指数增长。
- **对 wiki 的映射：** [pomdp](../../wiki/formalizations/pomdp.md)、[bayesian-belief-analysis](../../wiki/concepts/bayesian-belief-analysis.md)

### 4) Maybeck (1979) — 从 Bayes 估计到非线性滤波

- **来源：** P. S. Maybeck, *Stochastic Models, Estimation, and Control*, Vol. 1, Academic Press, 1979.
- **要点：**
  - 从 **Bayes 递推** 推导卡尔曼滤波；讨论非线性系统的 **线性化贝叶斯更新**（EKF 概率底座）。
  - 机器人读者常通过本书理解「belief = 后验分布」与「预测–校正」两步的贝叶斯含义。
- **对 wiki 的映射：** [bayesian-belief-analysis](../../wiki/concepts/bayesian-belief-analysis.md)、[kalman-filter](../../wiki/formalizations/kalman-filter.md)（亦见 [kalman_filter_ekf_primary_refs.md](./kalman_filter_ekf_primary_refs.md) §4）

### 5) Ghavamzadeh et al. (2015) — *Bayesian Reinforcement Learning: A Survey*

- **来源：** M. Ghavamzadeh, S. Mannor, J. Pineau, A. Tamar, *Bayesian Reinforcement Learning: A Survey*, Foundations and Trends in Machine Learning, 8(5–6):359–483, 2015. [作者 PDF](https://mohammadghavamzadeh.github.io/PUBLICATIONS/FoundationTrend-BRL.pdf)
- **要点：**
  - **Model-based BRL**：对转移/奖励参数设先验，用 **后验采样（PSRL）** 或 **Thompson sampling** 平衡探索–利用。
  - **Model-free BRL**：对价值函数或策略类设先验；讨论 **POMDP 中 belief** 与 **Bayes-adaptive MDP** 的关系。
  - 是 RL 视角下「贝叶斯分析」的 **标准综述入口**。
- **对 wiki 的映射：** [bayesian-belief-analysis](../../wiki/concepts/bayesian-belief-analysis.md)、[reinforcement-learning](../../wiki/methods/reinforcement-learning.md)

### 6) Poupart et al. (2006) / Ross et al. (2011) — Bayes-Adaptive POMDP

- **来源：**
  - N. Poupart, J. Vlassis, *Model-based Bayesian Reinforcement Learning in Partially Observable Domains*, Proc. Int. Symposium on Artificial Intelligence and Mathematics (ISAIM), 2006.
  - S. Ross, J. Pineau, B. Chaib-draa, P. Kreitmann, *A Bayesian Approach for Learning and Planning in Partially Observable Markov Decision Processes*, JMLR 12:1729–1770, 2011. [JMLR](https://jmlr.org/papers/v12/ross11a.html)
- **要点：**
  - 把 **模型不确定性** 与 **状态不确定性** 同时纳入 **Bayes-adaptive POMDP**；交互中 **学模型 + 跟踪 belief + 规划动作**。
  - 给出有限近似与 belief tracking 算法；代表「完全贝叶斯决策栈」的一手实现路线。
- **对 wiki 的映射：** [bayesian-belief-analysis](../../wiki/concepts/bayesian-belief-analysis.md)、[pomdp](../../wiki/formalizations/pomdp.md)

### 7) Tedrake, MIT Underactuated Robotics — Bayes 滤波讲义（机器人估计）

- **来源：** R. Tedrake, *Underactuated Robotics*, Ch. 16 Estimation. [课程站](https://underactuated.csail.mit.edu/estimation.html)
- **要点：**
  - 从 **Bayes 滤波** 推导 KF/EKF；讨论足式与操作机器人上的 **belief 传播** 与一致性。
  - 课程归档见 [mit_underactuated_kalman_lqr.md](../courses/mit_underactuated_kalman_lqr.md)。
- **对 wiki 的映射：** [state-estimation](../../wiki/concepts/state-estimation.md)、[bayesian-belief-analysis](../../wiki/concepts/bayesian-belief-analysis.md)

### 8) Sutton (2024) — *The One-Step Trap* 中对 Bayesian 分析的批判语境

- **来源：** R. S. Sutton, *The One-Step Trap*, 2024. [归档](../blogs/sutton_one_step_trap.md)
- **要点：**
  - 将 **POMDP / Bayesian 分析** 与一步模型 rollout 并列为「用单步结构拼长期预测」的高风险范式：belief 展开后 **计算树指数膨胀**。
  - 并非否定贝叶斯估计本身，而是警示 **把 belief 传播当作通用长期世界模型** 的可扩展性边界——与 [GVF](../../wiki/concepts/generalized-value-functions.md) 路线对照阅读。
- **对 wiki 的映射：** [bayesian-belief-analysis](../../wiki/concepts/bayesian-belief-analysis.md)、[generalized-value-functions](../../wiki/concepts/generalized-value-functions.md)

## 推荐继续阅读（外部）

- Silver & Veness (2010) POMCP — 蒙特卡洛 belief 树搜索
- Kochenderfer, *Decision Making under Uncertainty* — 航空航天 belief 规划教材

## 当前提炼状态

- [x] 贝叶斯分析八类一手来源摘录与 wiki 映射
- [x] 新建 `wiki/concepts/bayesian-belief-analysis.md` 并回写 `pomdp.md` 参考来源
- [ ] 后续可补：PSRL / Thompson sampling 算法专节
