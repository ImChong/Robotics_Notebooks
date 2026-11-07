# Sutton & Barto《Reinforcement Learning: An Introduction》第 2 版（MDP 与部分可观测章节通读）

> 来源归档（教材章节通读 · 支撑 `wiki/formalizations/mdp.md`、`wiki/concepts/embodied-rl-minimal-closed-loop.md`、`wiki/formalizations/pomdp.md`）

- **标题：** Reinforcement Learning: An Introduction, 2nd edition
- **作者：** Richard S. Sutton, Andrew G. Barto
- **类型：** book / rl / textbook
- **原文：** <http://incompleteideas.net/book/RLbook2020.pdf>（作者主页提供的 2020 版 PDF，约 548 页；2026-11-07 下载）；站点索引见 [`sites/incompleteideas-net-rich-sutton.md`](../sites/incompleteideas-net-rich-sutton.md)
- **入库日期：** 2026-11-07
- **一句话说明：** RL 标准教材；第 3 章给出有限 MDP 的标准记号（动力学函数 p(s′,r|s,a)、回报 G_t、v_π/q_π、Bellman 方程与最优方程），第 17.3 节给出"观测与状态"、部分可观测与 POMDP 的教材式处理。
- **核对范围：** 通读第 1 章 §1.1（RL 的定义、探索与利用、完整目标导向智能体）、第 3 章全部（§3.1 智能体–环境接口、§3.2 目标与奖励、§3.3 回报与折扣、§3.4 episodic / continuing 统一记号、§3.5 策略与价值函数、§3.6 最优策略与最优价值函数、§3.7 最优性与近似，含"倒立摆 / 回收机器人"例题与 Bellman 方程）以及 §17.3（观测与状态、历史、状态更新函数、POMDP 段落）。**没有读**第 2 章（多臂老虎机）、第 4–16 章（DP、MC、TD、函数逼近、策略梯度等）和第 17 章其余各节；本页引用的 PPO / SAC / Actor-Critic 内容均不以本书为依据。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **MDP 的定位（第 3 章开篇）：** MDP 是 RL 问题的数学理想化形式，动作影响的不只是即时奖励，还有后继状态与未来奖励，因此涉及延迟奖励与即时/延迟的权衡；bandit 估计 q*(a)，MDP 估计 q*(s,a) 或 v*(s)。
- **智能体–环境接口（§3.1）：** 离散时间 t=0,1,2,…，轨迹 S₀,A₀,R₁,S₁,A₁,R₂,…；有限 MDP 的动力学函数 p(s′,r|s,a)≐Pr{S_t=s′,R_t=r | S_{t−1}=s,A_{t−1}=a}，对每个 (s,a) 对 (s′,r) 求和为 1；由它可导出 p(s′|s,a)、r(s,a)、r(s,a,s′)。
- **马尔可夫性：** 原文称其"是对状态的限制，而非对决策过程的限制"——状态必须包含过去交互中对未来有影响的全部信息。
- **边界：** 智能体与环境的边界是"智能体可任意改变的范围"而非身体边界；机器人的电机、机械连杆、传感硬件应视为环境的一部分；奖励的计算也在环境一侧。这一节还说明可以把整个问题归约为三个信号：动作、状态、奖励。
- **奖励假设（§3.2）：** "目标与目的可以看作最大化所收到标量信号（奖励）累计和的期望值"。
- **回报（§3.3、§3.4）：** episodic 任务 G_t=R_{t+1}+…+R_T；折扣回报 G_t=Σ_{k≥0}γ^k R_{t+k+1}，0≤γ≤1；用"只转移到自身、奖励为 0 的吸收状态"统一 episodic 与 continuing 两种记号。例 3.4 把倒立摆既可当 episodic（每步 +1，回报为失败前的步数）也可当 continuing（失败时 −1、其余为 0，配合折扣）；练习 3.7 的迷宫机器人例子说明只在逃出时给 +1 且不折扣，会让智能体无从区分快慢。
- **价值函数与 Bellman 方程（§3.5）：** v_π(s)≐E_π[G_t|S_t=s]，q_π(s,a)≐E_π[G_t|S_t=s,A_t=a]；v_π(s)=Σ_aπ(a|s)Σ_{s′,r}p(s′,r|s,a)[r+γv_π(s′)]（式 3.14），v_π 是该方程的唯一解。
- **最优性（§3.6）：** 最优策略可能不唯一，共享同一 v*、q*；Bellman 最优方程 v*(s)=max_aΣ_{s′,r}p(s′,r|s,a)[r+γv*(s′)]；有 v* 则对 v* 贪心即最优，有 q* 则无需知道环境动力学即可选最优动作。
- **最优性与近似（§3.7）：** 直接求解最优方程依赖三条很少成立的假设——动力学精确已知、算力足够、状态具有马尔可夫性；作者指出"许多不同的决策方法可以看作对 Bellman 最优方程的近似求解"（并非"所有 RL 算法"；例如本书第 13 章的策略梯度并不是这样推出的，本次未读该章）。
- **部分可观测（§17.3）：** 函数逼近已经隐含地包含了部分可观测；明确处理需要四步：环境只给观测 O_t、用历史 H_t=A₀,O₁,…,O_t 定义状态、要求状态是历史的函数且可递归更新 S_{t+1}=u(S_t,A_t,O_{t+1})、并考虑计算可行性；POMDP 是其中一种取得马尔可夫状态的做法：假设存在从不对智能体可见的潜在状态 X_t，智能体维护对其的分布（belief）。原文同时批评一步预测迭代的误差会累积放大。

## 对 wiki 的映射

- [`wiki/formalizations/mdp.md`](../../wiki/formalizations/mdp.md)
- [`wiki/formalizations/pomdp.md`](../../wiki/formalizations/pomdp.md)
- [`wiki/concepts/embodied-rl-minimal-closed-loop.md`](../../wiki/concepts/embodied-rl-minimal-closed-loop.md)
- [`wiki/entities/sutton-barto-rl-book.md`](../../wiki/entities/sutton-barto-rl-book.md)

## 参考来源

- 教材 PDF：<http://incompleteideas.net/book/RLbook2020.pdf>
