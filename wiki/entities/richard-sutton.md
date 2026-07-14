---
type: entity
tags: [reinforcement-learning, theory, education, scaling-laws, model-based-rl, alberta]
status: complete
updated: 2026-07-14
related:
  - ./sutton-barto-rl-book.md
  - ../concepts/bitter-lesson.md
  - ../concepts/generalized-value-functions.md
  - ../concepts/bayesian-belief-analysis.md
  - ../methods/reinforcement-learning.md
  - ../methods/model-based-rl.md
  - ../formalizations/mdp.md
  - ../formalizations/bellman-equation.md
  - ../methods/intentional-updates-streaming-rl.md
  - ../concepts/embodied-scaling-laws.md
  - ../overview/robot-learning-overview.md
sources:
  - ../../sources/sites/incompleteideas-net-rich-sutton.md
  - ../../sources/blogs/sutton_bitter_lesson.md
  - ../../sources/blogs/sutton_one_step_trap.md
  - ../../sources/papers/generalized_value_functions_gvf_primary_refs.md
  - ../../sources/papers/bayesian_analysis_rl_primary_refs.md
summary: "Richard Sutton：RL 奠基人、2019 图灵奖得主；incompleteideas.net 托管 Sutton & Barto 教材、Alberta RL MOOC、Incomplete Ideas 博文与 RL 研究一手资料。"
---

# Richard Sutton

## 一句话定义

**Richard S. Sutton** 是现代 **强化学习（RL）** 的奠基研究者之一：与 Andrew Barto 合著 RL 标准教材、提出 TD learning / eligibility traces / options / Horde–GVF 等核心思想；其个人站 [incompleteideas.net](http://incompleteideas.net/) 是 RL 理论与 Alberta 学派的**一手资料总入口**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | Sutton 毕生主线：从交互中学习决策 |
| TD | Temporal-Difference Learning | Sutton 提出的 bootstrapping 价值学习族 |
| MDP | Markov Decision Process | RL 标准数学框架 |
| GVF | General Value Function | 预测任意 cumulant 的 value function 扩展 |
| MBRL | Model-Based Reinforcement Learning | Sutton 对「一步模型 rollout」持批判立场（见 One-Step Trap） |

## 为什么重要

- **教材与课程源头**：[*Reinforcement Learning: An Introduction*](./sutton-barto-rl-book.md) 与 UAlberta/Coursera RL 专项定义了全球 RL 入门话语；本库 [MDP](../formalizations/mdp.md)、[Bellman Equation](../formalizations/bellman-equation.md)、[Reinforcement Learning](../methods/reinforcement-learning.md) 均应以本站为权威外链。
- **Scaling 方法论作者**：[The Bitter Lesson](../concepts/bitter-lesson.md)（2019）系统论证 **search + learning** 随算力扩展胜过内置人类领域知识——早于具身 scaling 讨论，却与 [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md) 的「数据/算力共缩放」叙事形成层级对照。
- **MBRL 辩论的一手声音**：[One-Step Trap](../../sources/blogs/sutton_one_step_trap.md)（2024）批评 **单步转移模型 + rollout** 范式，主张 **options/GVF 时序抽象**——与当代「学世界模型再想象 rollout」路线直接交锋。

## 核心脉络（与机器人学习相关的子集）

### 1. 理论基础：TD 与 MDP 框架

- **Temporal-Difference learning**：用 bootstrapped 目标替代完整 Monte Carlo 回报，是在线 RL 的基石。
- **Eligibility traces**：把单步 TD 误差信用分配到历史状态；与 [GAE](../formalizations/gae.md)、[Intentional Updates（流式 RL）](../methods/intentional-updates-streaming-rl.md) 的 trace 几何一致。
- **函数逼近**：tile coding 等工具（[incompleteideas.net/tiles](http://incompleteideas.net/tiles/tiles3.html)）解决连续状态空间大表不可行问题。

### 2. 时序抽象：Options 与 Horde/GVF

- **Options framework**（1999）：在 MDP 与 semi-MDP 之间建立 temporal abstraction。
- **Horde**（2011）：无监督 sensorimotor 交互中并行学习大量 GVFs——详见 [Generalized Value Functions](../concepts/generalized-value-functions.md)。
- **Reward-respecting subtasks**（2023）：为 MBRL 提供时序抽象子任务——Sutton 眼中「一步陷阱」的解药方向。

### 3. 研究哲学（RLAI slogans 摘录）

| 口号 | 对机器人研究的含义 |
|------|-------------------|
| Take the agent's point of view | 从交互界面定义状态/动作/目标，而非工程师视角硬编码 |
| Don't ask the agent to know what it can't verify | 世界模型须可自验证，避免 brittle 手工知识库 |
| Experience is the data of AI | 真机/仿真交互数据是策略改进主燃料 |
| Work on ideas, not software | 与「堆工程 wrapper」相对，强调可复用的学习原理 |

### 4. 荣誉与机构

- **2019 ACM Turing Award**（与 Andrew Barto）
- UAlberta Computing Science 教授；Amii CIFAR AI Chair；Oak Lab / Openmind Research Institute 创始人

## 常见误区或局限

- **Bitter Lesson ≠ 否定一切领域知识**：短文强调的是 **长期可扩展的通用方法** 与 **短期人类先验** 的资源权衡，不是「仿真物理引擎无用」。
- **One-Step Trap 针对的是特定 MBRL 形式**：批评「不完美单步模型 + 长 horizon rollout」；不等于否定所有世界模型或模型辅助规划（如 Dyna、MPC+learned model 需分情形讨论）。
- **个人站链接会迁移**：教材官方页以 `incompleteideas.net/book/` 为准；第三方 PDF 镜像可能过时。

## 关联页面

- [Sutton & Barto RL 教材](./sutton-barto-rl-book.md)
- [Generalized Value Functions (GVFs)](../concepts/generalized-value-functions.md)
- [Bayesian Belief Analysis](../concepts/bayesian-belief-analysis.md)
- [The Bitter Lesson](../concepts/bitter-lesson.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Model-Based RL](../methods/model-based-rl.md)
- [MDP](../formalizations/mdp.md)
- [Bellman Equation](../formalizations/bellman-equation.md)
- [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)
- [Robot Learning Overview](../overview/robot-learning-overview.md)

## 参考来源

- [incompleteideas.net 一手资料索引](../../sources/sites/incompleteideas-net-rich-sutton.md)
- [GVF 一手资料索引](../../sources/papers/generalized_value_functions_gvf_primary_refs.md)
- [贝叶斯分析一手资料索引](../../sources/papers/bayesian_analysis_rl_primary_refs.md)
- [The Bitter Lesson 原始资料](../../sources/blogs/sutton_bitter_lesson.md)
- [The One-Step Trap 原始资料](../../sources/blogs/sutton_one_step_trap.md)

## 推荐继续阅读

- [incompleteideas.net](http://incompleteideas.net/) — 教材、MOOC、FAQ、演讲总索引
- [The Bitter Lesson](http://incompleteideas.net/IncIdeas/BitterLesson.html) — scaling 方法论原文
- [The One-Step Trap](http://incompleteideas.net/IncIdeas/OneStepTrap.html) — MBRL 一步模型批判
- [RL FAQ](http://incompleteideas.net/RL-FAQ.html) — RL 入门 FAQ（2004）
- [Coursera: Reinforcement Learning Specialization](https://www.coursera.org/specializations/reinforcement-learning) — Alberta RL MOOC
- [ACM Turing Award video](https://www.youtube.com/watch?v=RrXibq7-W6o) — RL 科普
