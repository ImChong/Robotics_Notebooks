---
type: entity
tags: [reinforcement-learning, education, textbook, theory]
status: complete
updated: 2026-07-14
related:
  - ./richard-sutton.md
  - ../methods/reinforcement-learning.md
  - ../formalizations/mdp.md
  - ../formalizations/bellman-equation.md
  - ../formalizations/gae.md
  - ../methods/model-based-rl.md
  - ../entities/hands-on-rl-book.md
  - ../../roadmap/depth-rl-locomotion.md
sources:
  - ../../sources/sites/incompleteideas-net-rich-sutton.md
summary: "Sutton & Barto《Reinforcement Learning: An Introduction》是 RL 领域标准教材；官方电子版、习题与教学材料托管于 incompleteideas.net。"
---

# Sutton & Barto RL 教材

## 一句话定义

**Reinforcement Learning: An Introduction**（Richard S. Sutton & Andrew G. Barto）是强化学习领域的**标准教材**：以 MDP 为数学框架，系统讲解动态规划、Monte Carlo、TD learning、函数逼近、策略梯度与规划等核心内容。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 教材主题：从交互中学习决策 |
| MDP | Markov Decision Process | 第 3 章有限 MDP 框架 |
| TD | Temporal-Difference Learning | 第 6–7 章 TD(0)/TD($\lambda$) 与 eligibility traces |
| MC | Monte Carlo Methods | 第 5 章基于完整 episode 的学习 |
| DP | Dynamic Programming | 第 4 章基于模型的 Bellman 备份 |

## 为什么重要

- **理论参照系**：本库 [MDP](../formalizations/mdp.md)、[Bellman Equation](../formalizations/bellman-equation.md)、[GAE](../formalizations/gae.md)、[Reinforcement Learning](../methods/reinforcement-learning.md) 的符号与叙述均与此书对齐。
- **与中文/工程教材互补**：[动手学强化学习（蘑菇书）](./hands-on-rl-book.md) 偏 PyTorch 实践；本书偏 **原理与证明脉络**；[depth-rl-locomotion](../../roadmap/depth-rl-locomotion.md) Stage 0 可与 Spinning Up 并列推荐。
- **一手版本托管**：第 2 版（2018, MIT Press）官方 PDF、errata、slides、代码解答交换均在 [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)。

## 版本与获取

| 版本 | 出版 | 官方页 |
|------|------|--------|
| 第 2 版 | MIT Press, 2018 | <http://incompleteideas.net/book/the-book-2nd.html> |
| 第 1 版 | MIT Press, 1998 | <http://incompleteideas.net/book/first/the-book.html> |

配套资源：全书 PDF（无页边距版）、分章 errata、slides、LaTeX 记号 `.sty`、习题官方解答（教师/自学申请）。

## 章节与机器人学习的映射（选读）

| 章节主题 | 本库对应页 |
|----------|-----------|
| Ch.3 有限 MDP | [MDP](../formalizations/mdp.md) |
| Ch.4 动态规划 | [Bellman Equation](../formalizations/bellman-equation.md) |
| Ch.6–7 TD / traces | [GAE](../formalizations/gae.md)、[Intentional Updates](../methods/intentional-updates-streaming-rl.md) |
| Ch.8 函数逼近 | 人形 loco 中大状态空间策略网络 |
| Ch.9 on-policy 预测与控制 | PPO/Actor-Critic 理论背景 |
| Ch.17 规划与学习 | [Model-Based RL](../methods/model-based-rl.md) |

## 常见误区或局限

- **不是工程手册**：不含 Isaac Lab / MuJoCo 部署细节；需搭配 [蘑菇书](./hands-on-rl-book.md) 或 Spinning Up。
- **函数逼近章节的 caution**：Q-learning + 线性函数逼近不 sound 等结论对深度 RL 仍有警示意义，但不能直接外推到所有神经网络设定。
- **外链 PDF 镜像**：优先使用 incompleteideas.net 官方页，避免过时 third-party 镜像。

## 关联页面

- [Richard Sutton](./richard-sutton.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [MDP](../formalizations/mdp.md)
- [动手学强化学习（蘑菇书）](./hands-on-rl-book.md)
- [RL Locomotion 纵深路线](../../roadmap/depth-rl-locomotion.md)

## 参考来源

- [incompleteideas.net 一手资料索引](../../sources/sites/incompleteideas-net-rich-sutton.md)

## 推荐继续阅读

- [Sutton & Barto 第 2 版官方页](http://incompleteideas.net/book/the-book-2nd.html)
- [Alberta RL Coursera 专项](https://www.coursera.org/specializations/reinforcement-learning) — 教材配套 MOOC
- [RL FAQ](http://incompleteideas.net/RL-FAQ.html) — Sutton 推荐的入门路径与 FAQ
- [OpenAI Spinning Up](https://spinningup.openai.com/) — 工程向补充
