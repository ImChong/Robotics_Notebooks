# Sutton & Barto RL 教材 · 第 1 章 §1.6 强化学习史

- **类型**：course / textbook excerpt
- **标题**：1.6 History of Reinforcement Learning
- **原始链接**：<http://incompleteideas.net/book/ebook/node12.html>
- **所属教材**：[Sutton & Barto, *Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html)（第 1 版 LaTeX2HTML 在线版；第 2 版 PDF 第 1 章对应小节）
- **收录日期**：2026-08-30
- **抓取说明**：以 **2026-08-30** 对 `incompleteideas.net/book/ebook/node12.html` 公开 HTML 为准；HTTPS 证书异常，HTTP 可访问。

## 一句话

Sutton & Barto 用 **三条独立又交织的研究脉络**（试错学习、最优控制、时序差分）梳理现代 RL 如何从心理学、动态规划与早期 AI 实验中汇合而成。

## 为什么值得保留

- **本库 RL 方法页的史学锚点**：[`wiki/methods/reinforcement-learning.md`](../../wiki/methods/reinforcement-learning.md)、[`wiki/formalizations/mdp.md`](../../wiki/formalizations/mdp.md)、[`wiki/formalizations/bellman-equation.md`](../../wiki/formalizations/bellman-equation.md) 的符号与算法谱系均可回溯到本节人物与里程碑。
- **厘清常见混淆**：监督学习 vs 试错学习、model-based 最优控制 vs 无模型 RL、secondary reinforcer vs TD 学习——避免把「用误差更新权重」误当 trial-and-error。
- **机器人读者选读**：Actor–Critic 源于 1983 杆平衡实验；Q-learning（1989）与 TD-Gammon（1992）是深度 RL 游戏里程碑（见 [`wiki/concepts/deep-rl-game-milestones.md`](../../wiki/concepts/deep-rl-game-milestones.md)）的直接前史。

## 三条主线（原文结构）

### 1. 最优控制 / 动态规划线（通常不含学习）

| 年代 | 人物 / 工作 | 要点 |
|------|-------------|------|
| 1950s 中 | Richard Bellman 等 | Hamilton–Jacobi 延伸 → **value function**、**Bellman 方程** |
| 1957 | Bellman | **动态规划（DP）**；离散随机情形 → **MDP** |
| 1960 | Ron Howard | MDP **策略迭代** |
| 此后 | Bertsekas、Puterman 等 | 部分可观测 MDP、异步 DP、近似 DP；**维度灾难** |

Sutton & Barto 立场：凡能有效求解 RL 问题的方法都算 RL；DP 虽需完整模型，但其**增量迭代**与学习方法同族，应与不完全知识情形一并讲授。

### 2. 试错学习线（心理学 → 早期 AI）

| 年代 | 人物 / 工作 | 要点 |
|------|-------------|------|
| 1911 | Edward Thorndike | **效果律（Law of Effect）**：好/坏结果改变动作再选倾向；**选择 + 联想** = search + memory |
| 1954 | Minsky；Farley & Clark | 最早计算化试错学习；Minsky **SNARC**；Minsky (1961) **credit assignment** |
| 1960s–70s | 名义「RL」实为监督学习 | Rosenblatt、Widrow–Hoff 等用奖惩语言但做模式识别 → 真试错研究一度稀少 |
| 1963+ | John Andreae **STeLLA** | 与环境交互试错 + 内部世界模型；影响有限 |
| 1960s–80s | **Learning automata** / 多臂老虎机 | Tsetlin；Barto & Anandan (1985) 扩展到 associative |
| 1975+ | John Holland | 选择原则；1986 **classifier systems**（含 GA + value） |
| 1972–82 | **Harry Klopf** | 复兴试错线；hedonic drive；影响 Barto & Sutton 区分监督 vs RL |
| 1981–83 | Barto, Sutton, Anderson | Actor–Critic + TD 用于 **杆平衡**；现代 RL 复兴核心 |

### 3. 时序差分（TD）线（三线交汇的胶水）

| 年代 | 人物 / 工作 | 要点 |
|------|-------------|------|
| 1954 | Minsky | **次级强化物** 思想或影响人工学习 |
| 1959 | Arthur Samuel | 跳棋程序；**TD 思想**在线改评估函数（受 Shannon 象棋评估启发） |
| 1972 | Klopf | generalized reinforcement / 局部强化 |
| 1978–81 | Sutton | 时序 successive prediction；与 Barto 经典条件反射心理模型 |
| 1977 | Ian Witten | 最早发表的 **tabular TD(0)** 规则（MDP 自适应控制） |
| 1981 | Barto, Sutton, Anderson | **Actor–Critic** 架构 |
| 1988 | Sutton | TD 与 control 分离；**TD(λ)** 与收敛性 |
| **1989** | **Chris Watkins** | **Q-learning** — 三线正式汇合 |
| 1992 | Gerry Tesauro | **TD-Gammon** 西洋双陆棋 |

## 核心论点摘录

1. **现代 RL = 三线于 1980 年代末汇合**，而非单一学科发明。
2. **试错学习的本质**是 selectional（试选动作）+ associative（与情境绑定），监督学习只有后者。
3. **Credit assignment**（功劳分配）是 RL 算法族共同面对的问题。
4. **TD 学习**由 successive estimates 之差驱动，与动物学习「次级强化」有概念渊源，但是 RL 特有方法。
5. 教材将 **DP / 已知模型最优控制** 与 **不完全知识 RL** 视为同一主题的两端。

## 对 wiki 的映射

- 新建：[wiki/concepts/reinforcement-learning-history.md](../../wiki/concepts/reinforcement-learning-history.md)
- 交叉更新：[wiki/entities/sutton-barto-rl-book.md](../../wiki/entities/sutton-barto-rl-book.md)、[wiki/methods/reinforcement-learning.md](../../wiki/methods/reinforcement-learning.md)
- 关联：[wiki/concepts/cartpole.md](../../wiki/concepts/cartpole.md)（1983 杆平衡）、[wiki/entities/richard-sutton.md](../../wiki/entities/richard-sutton.md)

## 推荐继续阅读（外部）

- [Sutton & Barto 第 2 版官方页](http://incompleteideas.net/book/the-book-2nd.html)
- [§1.6 在线 HTML](http://incompleteideas.net/book/ebook/node12.html)
- Bryson (1996) — 最优控制权威史（原文引用）
