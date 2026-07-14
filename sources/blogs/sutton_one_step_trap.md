# Rich Sutton — The One-Step Trap

- **类型**：blog / essay（Incomplete Ideas；原为 X 帖文整理）
- **作者**：Richard S. Sutton
- **原始链接**：<http://incompleteideas.net/IncIdeas/OneStepTrap.html>
- **发布日期**：2024-07-18
- **收录日期**：2026-07-14
- **站点索引**：[incompleteideas-net-rich-sutton.md](../sites/incompleteideas-net-rich-sutton.md)

## 一句话

**一步陷阱**：误以为智能体的大部分学习预测都可以是 **一步预测**，再通过迭代 rollout 得到长期预测——在随机世界与实践中，单步误差复合、计算复杂度指数爆炸，使该路线不可行。

## 为什么值得保留

- **直接批评主流 MBRL / 世界模型叙事**：许多系统学 one-step transition model 再 rollout，Sutton 认为这在非完美单步预测下 **误差累积** 且 **计算不可行**（随机策略下未来是概率树，复杂度对预测长度指数级）。
- **给出 Alberta 学派替代路线**：时序抽象模型——**options + GVFs**（General Value Functions），并引用 Options framework、Horde、Reward-respecting subtasks 等一手论文。
- 与本站 [Model-Based RL](../../wiki/methods/model-based-rl.md)、[Foundation Policy](../../wiki/concepts/foundation-policy.md) 中「隐空间世界模型 + TD」路线形成 **方法论对照**。

## 核心论点

| 主张 | 说明 |
|------|------|
| 一步模型的诱惑 | 若单步预测完美，迭代可得完美长期预测——含一粒真理 |
| 实践失败模式 | 单步误差 compound；长期预测质量差 |
| 计算障碍 | 随机环境下需对指数级分支加权，rollout 一般 infeasible |
| 广泛误用 | POMDP、Bayesian 分析、控制论、压缩理论 AI 中常见 |
| Sutton 主张的解 | 用 **options 与 GVFs** 构建时序抽象世界模型 |

### 引用论文（原文列出）

- Sutton, Precup, Singh (1999). *Between MDPs and semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning.*
- Sutton et al. (2011). *Horde: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction.*
- Sutton et al. (2023). *Reward-respecting subtasks for model-based reinforcement learning.*

## 对 wiki 的映射

- [wiki/entities/richard-sutton.md](../../wiki/entities/richard-sutton.md)
- 交叉：[wiki/methods/model-based-rl.md](../../wiki/methods/model-based-rl.md)

## 参考链接

- 原文：<http://incompleteideas.net/IncIdeas/OneStepTrap.html>
