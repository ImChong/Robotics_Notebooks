---
type: formalization
tags: [rl, mdp, markov, decision-process]
status: complete
related:
  - ./bellman-equation.md
  - ../methods/reinforcement-learning.md
  - ../methods/inverse-reinforcement-learning.md
  - ../methods/policy-optimization.md
  - ../methods/ppo.md
  - ../concepts/embodied-rl-minimal-closed-loop.md
  - ../concepts/cartpole.md
original:
  - ../../sources/books/sutton_barto_rl_intro_2nd_ed.md
sources:
  - ../../sources/books/sutton_barto_rl_intro_2nd_ed.md
  - ../../sources/papers/policy_optimization.md
  - ../../sources/blogs/wechat_robotshub_ppo_locomotion_fundamentals.md
summary: "马尔可夫决策过程（MDP）：用状态、动作、转移动力学、奖励与折扣描述序贯决策的标准框架，及回报、价值函数、Bellman 方程与最优性；是强化学习的数学基础。"
status: complete
updated: 2026-11-07
---

# Markov Decision Process (MDP)

**马尔可夫决策过程**：在离散时间步中，智能体根据当前状态选择动作，环境根据转移概率回应新状态和奖励的数学框架，是强化学习的理论基础。

## 一句话定义

>"智能体在不确定环境中做决策——每一步选动作，获得奖励，下一步到达哪个状态由转移概率决定"——这一过程的数学形式化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MDP | Markov Decision Process | 状态–动作–奖励–转移的标准序贯决策建模框架 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| IMU | Inertial Measurement Unit | 惯性测量单元，提供加速度与角速度 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| OCP | Optimal Control Problem | MPC 每步求解的有限时域最优控制问题 |
| Reward | Reward Function | 塑造强化学习策略行为的标量反馈 |
| POMDP | Partially Observable Markov Decision Process | 部分可观测的 MDP，部署时观测受限的常见建模 |

## 为什么重要

MDP 是 RL 的根基。RL 中所有算法——Policy Gradient、SARSA、Q-learning——本质上都是在解 MDP。

不懂 MDP，就不懂为什么 RL 要"最大化累积折扣奖励"，也不知道为什么值函数、策略、模型这些概念会出现。

## 核心原理：五元组、回报与价值函数

> 记号说明：本页的五元组 $(S,A,P,R,\gamma)$ 是常见的教学写法。Sutton & Barto 教材（第 3 章）用一个四参数的**动力学函数** $p(s',r\mid s,a)\doteq\Pr\{S_t=s',R_t=r\mid S_{t-1}=s,A_{t-1}=a\}$ 同时刻画下一状态与奖励，对每个 $(s,a)$ 有 $\sum_{s',r}p(s',r|s,a)=1$；由它可导出 $p(s'|s,a)$、$r(s,a)$、$r(s,a,s')$，两种写法等价。教材的轨迹下标约定为 $S_0,A_0,R_1,S_1,A_1,R_2,\dots$（奖励 $R_{t+1}$ 与 $S_{t+1}$ 同时产生）。

**马尔可夫性是对「状态」的要求，不是对过程的限制**：教材原话大意为状态必须包含过去交互中所有对未来有影响的信息；若不满足，就应改用 [POMDP](./pomdp.md) 或把历史并入状态。

### 五元组

$$M = (S, A, P, R, \gamma)$$

| 符号 | 含义 | 人形机器人例子 |
|------|------|----------------|
| $S$ | 状态空间 | 关节角度、角速度、IMU、接触状态 |
| $A$ | 动作空间 | 各关节力矩/位置指令 |
| $P(s'|s,a)$ | 转移概率 | 给定当前状态和动作，下一状态是什么 |
| $R(s,a,s')$ | 奖励函数 | 走得稳给正奖励、摔倒给负奖励 |
| $\gamma \in [0,1]$（持续任务需 $<1$） | 折扣因子 | 近期奖励重要还是远期奖励重要；有效视野约 $1/(1-\gamma)$ **步** |

### 轨迹

一个完整的交互序列（episode）：

$$s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2 \xrightarrow{a_2} \cdots \xrightarrow{a_{T-1}} s_T$$

累积折扣奖励：

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

教材还用「只转移到自身、奖励恒为 0 的**吸收状态**」把有终点的 episodic 任务与无终点的 continuing 任务统一到同一记号。

恒定奖励 $r$ 时，$G$ 的尺度约 $r/(1-\gamma)$，常把 $1/(1-\gamma)$ 叫**有效视野**（多少步之后的奖励可忽略）。这是**步数**不是秒：控制频率升高时，同样 $\gamma$ 覆盖的真实时间变短。50 Hz 下 $\gamma=0.99$ 大约 2 s；200 Hz 下只剩约 0.5 s。运控里随控制频率一起改 $\gamma$，否则策略会「变短视」。详见 [PPO](../methods/ppo.md)。

### 策略 $\pi$

策略 $\pi(a|s)$ 是在状态 $s$ 下选择动作 $a$ 的概率分布：

$$\pi: S \rightarrow \text{Prob}(A)$$

- **确定性策略**：$a = \pi(s)$，输出单一动作
- **随机策略**：$\pi(a|s)$，输出动作分布（更适合部分可观测或高维连续动作空间）

### 值函数

**状态值函数** $V^\pi(s)$：从状态 $s$ 开始，按策略 $\pi$ 行事的期望累积折扣奖励

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid s_t = s\right]$$

**动作值函数** $Q^\pi(s,a)$：从状态 $s$ 出发，执行动作 $a$ 后按策略 $\pi$ 行事的期望累积折扣奖励

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid s_t = s, a_t = a\right]$$

两者关系：

$$V^\pi(s) = \sum_a \pi(a|s) \, Q^\pi(s,a)$$

### 最优性

最优策略 $\pi^*$ 满足：

$$V^{\pi^*}(s) \geq V^\pi(s), \quad \forall s, \pi$$

等价地：

$$Q^{\pi^*}(s,a) \geq Q^\pi(s,a), \quad \forall s,a$$

最优策略可能不唯一，但它们共享同一个最优值函数 $V^*$ 和 $Q^*$（教材 §3.6）。

## Bellman 方程

值函数满足递归关系（Bellman 方程）：

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \bigl( R(s,a,s') + \gamma V^\pi(s') \bigr)$$

最优值函数（Bellman 最优方程）：

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \bigl( R(s,a,s') + \gamma V^*(s') \bigr)$$

$V^\pi$ 是 Bellman 方程的**唯一解**；若动力学已知，解 Bellman 最优方程得到 $V^*$，再对 $V^*$ 做一步贪心即是最优策略。有 $Q^*$ 则连动力学都不需要，直接取 $\arg\max_a Q^*(s,a)$。

## 人形机器人中的 MDP

| MDP 元素 | 人形 locomotion 具体化 |
|----------|------------------------|
| 状态 $s$ | $(q, \dot{q}, r, \dot{r}, c, \omega)$ — 关节位置/速度、基座位置/速度、质心位置、角速度 |
| 动作 $a$ | 关节力矩指令或位置指令 |
| 转移 $P$ | 物理仿真器（MuJoCo/IsaacGym）给出 |
| 奖励 $R$ | 行走速度 + 稳定性 + 能耗 + 摔倒惩罚的加权组合 |
| 折扣 $\gamma$ | 0.99（PPO 论文 MuJoCo 设置）；0.995 等取值为社区经验（推测，未逐项核对）|

典型的 RL 训练就是通过与仿真器交互，隐式地解这个 MDP 的 Bellman 最优方程。

## 和 RL 的关系

```
MDP（数学框架）
    ↓ 精确解（已知模型）→ 动态规划 → Value Iteration / Policy Iteration
    ↓ 近似解（模型未知）→ RL → Q-learning / Policy Gradient / Actor-Critic
```

教材 §3.7 的原话是「许多」决策方法可看作对 Bellman 最优方程的近似求解（用真实经历的转移代替期望转移）；并不是每一种 RL 算法都如此推出——例如直接对策略参数做梯度上升的策略梯度类方法，其推导不经过 Bellman 最优方程（本次入库未读教材第 13 章，仅作提示）。

## 工程实践

- **边界划在哪里：** 教材强调智能体—环境的边界是「智能体能任意改变的范围」，不是机器人的物理外壳。机器人的电机、连杆、传感硬件都应算环境；奖励的计算也在环境一侧。对应到仿真训练：策略只输出关节目标或力矩，PD 控制器、执行器模型、接触物理和奖励函数都属于「环境」。
- **episodic 还是 continuing：** 教材以倒立摆为例——可按 episodic（每步 $+1$，回报等于失败前步数）或 continuing（失败时 $-1$、其余为 0 并配合折扣）处理，两者都鼓励尽量久地保持平衡。工程上需要区分「真终止」与「超时截断」，前者不 bootstrap，后者应 bootstrap；见 [Cartpole](../concepts/cartpole.md)。
- **奖励要能表达目标：** 教材练习 3.7 的迷宫例子说明，只在逃出迷宫时给 $+1$、且不折扣、也不加时间惩罚，会让智能体无法区分「快」和「慢」。奖励设计见 [Reward Design](../concepts/reward-design.md)。
- **最小闭环：** 把 $S,A,P,R,\gamma$ 逐项落成可运行的 reset/step 接口，见 [具身 RL 最小闭环](../concepts/embodied-rl-minimal-closed-loop.md)。

## 局限与风险

- **直接解 Bellman 最优方程很少可行：** 教材指出它依赖三条假设——环境动力学精确已知、算力足以完成求解、状态具有马尔可夫性；实际中往往至少一条不成立（教材以约 $10^{20}$ 个状态的西洋双陆棋为例说明算力不足），所以 RL 只能近似求解。
- **马尔可夫性在真机上常不成立：** 传感噪声、延迟、遮挡使观测不是完整状态，需要 [POMDP](./pomdp.md) 视角、历史堆叠或循环网络。
- **有效视野按步数而非秒：** $1/(1-\gamma)$ 以控制步计，控制频率改变时 $\gamma$ 覆盖的真实时间随之变化（见上文）。
- **表格理论 vs 深度 RL：** Bellman 方程的唯一解、策略改进单调等结论是对有限 MDP 成立的；深度 RL 中的函数逼近下一般没有这些保证。
- **「奖励假设」本身是建模选择：** 把一切目标都压成标量奖励的累计，是教材明确承认的抽象，不一定适合所有任务（例如多目标、约束，见 [CMDP](./cmdp.md)）。

## 参考来源

- [Sutton & Barto 教材归档（第 3 章、§17.3 通读）](../../sources/books/sutton_barto_rl_intro_2nd_ed.md) — 动力学函数、回报、Bellman 方程、最优性与近似
- [RobotsHub：万字解析运控 PPO](../../sources/blogs/wechat_robotshub_ppo_locomotion_fundamentals.md) — 有效视野按步数计、与控制频率耦合
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — MDP 与 RL 标准教材，Chapter 3（原文：<http://incompleteideas.net/book/RLbook2020.pdf>）
- Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* — MDP 理论权威参考
- Bellman, *Dynamic Programming* (1957) — MDP 理论基础

## 关联页面

- [Reinforcement Learning](../methods/reinforcement-learning.md) — MDP 是 RL 的理论根基
- [Inverse Reinforcement Learning](../methods/inverse-reinforcement-learning.md) — 已知（近）最优行为，反推 MDP 里缺失的 $R$
- [PPO](../methods/ppo.md) — 折扣 $\gamma$ 的有效视野随控制频率变化
- [Optimal Control](../concepts/optimal-control.md) — OCP 和 MDP 的关系：OCP 是确定性的，RL 是随机性的；两者都是"最优决策"问题
- [Reward Design](../concepts/reward-design.md) — MDP 中奖励函数的设计直接决定学到的策略
- [Cartpole 问题](../concepts/cartpole.md) — 四维完全可观测 MDP 的最小对照：Gym 失败信号 vs Isaac shaping
- [POMDP](./pomdp.md) — MDP 加观测模型的扩展，真实机器人系统的更准确框架
- [Bellman 方程](./bellman-equation.md) — MDP 最优策略通过 Bellman 最优方程表达
- [CMDP](./cmdp.md) — 带约束的 MDP 扩展
- [具身 RL 最小闭环](../concepts/embodied-rl-minimal-closed-loop.md) — 把五元组落成可运行仿真循环

## 推荐继续阅读

- [Spinning Up — Part 1: Key Concepts of RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [Sutton & Barto — Chapter 3: Finite MDPs](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
