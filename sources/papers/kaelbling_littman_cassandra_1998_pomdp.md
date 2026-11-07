# Planning and Acting in Partially Observable Stochastic Domains（Kaelbling, Littman, Cassandra, 1998）

> 来源归档（原文通读 · 支撑 `wiki/formalizations/pomdp.md`）

- **标题：** Planning and acting in partially observable stochastic domains
- **作者：** Leslie Pack Kaelbling, Michael L. Littman, Anthony R. Cassandra
- **类型：** paper / planning / POMDP
- **出处：** *Artificial Intelligence* 101 (1998) 99–134；PII S0004-3702(98)00023-X
- **原文：** <https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf>（作者主页提供的 PDF，2026-11-07 下载，约 10 页重排版）
- **入库日期：** 2026-11-07
- **一句话说明：** 把运筹学里的 MDP / POMDP 理论引入 AI 规划：用 belief state 把 POMDP 转成连续状态的 belief MDP，证明有限步最优价值函数是分段线性凸函数，并提出精确求解的 witness 算法。
- **核对范围：** 通读摘要与 §1 引言、§2 MDP（含折扣、策略、价值迭代与 Bellman 误差界）、§3 部分可观测（POMDP 定义、状态估计器 SE、belief 更新公式、belief MDP）、§4 价值函数（策略树、α 向量、分段线性凸、witness 算法概要）、§5.1 tiger 问题的设定与有限步策略开头、§6.1–§6.5 相关工作的前半、§7 扩展与结论，以及附录 A 定理 A.1 的陈述与证明开头。**没有读**§4.4 witness 算法的全部实现细节与线性规划、§5.2–§5.4 其余图表分析、§6.6 之后的相关工作；定理证明未逐步重推。没有运行任何求解器。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **POMDP 定义（§3.1）：** 六元组 ⟨S,A,T,R,Ω,O⟩；T:S×A→Π(S)，R:S×A→ℝ，Ω 为有限观测集，O:S×A→Π(Ω) 给出"执行动作 a 并落入状态 s′ 之后观测到 o"的概率 O(s′,a,o)。原文脚注 4 指出也可建模成观测依赖前一状态，表达力等价，只是需扩充状态空间。原文 MDP 折扣 0<γ<1。
- **结构（§3.2、图 2）：** POMDP 智能体分解为状态估计器 SE 与策略 π；SE 根据上一 belief、最近动作与当前观测更新 belief，π 以 belief 而非世界状态为输入。belief 是状态的概率分布，是历史与初始 belief 的充分统计量，因此 belief 过程是马尔可夫的；原文强调"最可能的状态"不足以作为 belief，因为智能体需要考虑自己的不确定性。
- **belief 更新（§3.3）：** b′(s′)=O(s′,a,o)·Σ_s T(s,a,s′)b(s) / Pr(o|a,b)，分母是归一化因子。
- **belief MDP（§3.4）：** 状态空间为 belief 集合 B，转移 τ(b,a,b′)=Σ_o Pr(b′|b,a,o)Pr(o|a,b)，奖励 ρ(b,a)=Σ_s b(s)R(s,a)；该连续空间 MDP 的最优策略配合正确的状态估计器即给出原 POMDP 的最优行为（折扣无限时域意义下）。
- **价值函数结构（§4）：** t 步策略树（节点为动作、边为观测），V_p(b)=b·α_p，V_t(b)=max_{p}b·α_p，因此有限步最优价值函数是 belief 空间上的分段线性凸函数；无限时域折扣价值函数仍是凸的，但可能有无穷多"面"，可以被任意逼近。
- **算法与复杂度（§4.4、§7）：** witness 算法精确地逐步做价值迭代（每个动作的 Q 函数用 witness 点逐个补全策略树）；原文证明单次价值迭代的运行时间关于 |S|、|A|、|Ω| 以及前一轮价值函数与当前各 Q 函数所含策略树的数量为多项式（而这些策略树数量本身可以指数级增大），但实验表明即使 witness 算法也在 |S|>15 且 |Ω|>15 的中等规模问题上变得不实用；作者当时在探索函数逼近与基于仿真的方法，并报告了一个 89 状态、16 观测的走廊导航实例的良好近似解。
- **tiger 例子（§5.1）：** 两扇门、一边是老虎；LISTEN 代价 −1 且观测只有 85% 准确，开错门 −100、开对 +10；开门后问题重置；这是展示"信息收集动作与改变世界的动作在 POMDP 里不作区分"的标准例。
- **与传统规划的区别（§1、§6.1）：** POMDP 不区分"为改变世界的动作"与"为获取信息的动作"，两者通过观测函数统一；MDP 与经典规划假设执行时状态完全已知；完全可观与完全不可观都是 POMDP 的特例。
- **对确定性观测-动作映射的评价（§3）：** 把最近一次观测直接映射成动作、不记忆历史，对外观相同的不同位置会做同样的事，效果有限；带随机性的观测到动作映射可以更好，但确定性映射容易陷入循环；真正有效的做法需要用历史记忆消歧。
- **局限（原文自述）：** 方法要求状态被枚举表示，不是组合式表示；需要已知完整正确的世界模型；精确求解只适用于小规模；模型获取（学习 POMDP 模型）不在本文范围。

## 对 wiki 的映射

- [`wiki/formalizations/pomdp.md`](../../wiki/formalizations/pomdp.md)
- [`wiki/formalizations/mdp.md`](../../wiki/formalizations/mdp.md)

## 参考来源

- 论文：<https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf>
- 期刊 DOI：<https://doi.org/10.1016/S0004-3702(98)00023-X>
