---
type: concept
tags: [reinforcement-learning, prediction, horde, model-based-rl, temporal-abstraction, alberta]
status: complete
related:
  - ../entities/richard-sutton.md
  - ../methods/model-based-rl.md
  - ../methods/reinforcement-learning.md
  - ../formalizations/mdp.md
  - ../formalizations/bellman-equation.md
  - ./bayesian-belief-analysis.md
  - ../methods/intentional-updates-streaming-rl.md
sources:
  - ../../sources/papers/generalized_value_functions_gvf_primary_refs.md
  - ../../sources/blogs/sutton_one_step_trap.md
summary: "广义价值函数（GVF）用策略条件的折扣 cumulant 累积定义预测性知识，Horde 架构可并行 off-policy 学习数千路长期预测。"
updated: 2026-07-14
---

# Generalized Value Functions (GVFs)

**广义价值函数（GVF）**：在标准 value function 框架下，把「奖励折扣和」推广为「**任意 cumulant 信号** 在 **策略 π** 与 **终止/折扣 γ** 下的期望累积」——每条 GVF 即一个 grounded 的预测问题，可用 TD 族算法 **span-independent** 地在线学习。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GVF | Generalized Value Function | 预测任意 cumulant 的策略条件价值函数 |
| TD | Temporal-Difference Learning | 用 bootstrapped 目标在线学习 value/GVF |
| RL | Reinforcement Learning | GVF 源于 RL 的 value function 语义与算法 |
| MBRL | Model-Based Reinforcement Learning | Sutton 主张用 GVF+options 替代 naive 一步模型 rollout |
| Horde | — | 并行学习大量 GVF 的 Alberta 架构（AAMAS 2011） |

## 为什么重要

- **预测性世界知识**：机器人需回答「若继续当前行为，传感器/接触/位姿信号未来如何累积」——这类问题不总是任务奖励，但直接影响规划与异常检测。
- **可学习且 grounded**：GVF 的「真值」是交互中可验证的期望累积；比纯符号规则更易从 **无监督 sensorimotor 数据** 学得（Horde 核心主张）。
- **与一步模型辩论直接相关**：[Richard Sutton](../entities/richard-sutton.md) 在 [*One-Step Trap*](../../sources/blogs/sutton_one_step_trap.md) 中把 **options + GVFs** 作为 **单步转移模型长 horizon rollout** 的替代——学多路 **直接长期预测**，而非迭代复合单步误差。

## 核心原理

### GVF 四元组（question）

每条 GVF 由以下要素定义（Horde 2011）：

| 要素 | 符号 | 含义 |
|------|------|------|
| 策略 | $\pi(a\|s)$ | 条件预测所依据的行为策略 |
| Cumulant | $c(s)$ 或 $C_{t+1}$ | 被累积的伪奖励信号（传感器读数、事件指示等） |
| 终止/折扣 | $\gamma(s)\in[0,1]$ | 每步终止概率 $1-\gamma(s)$ 或折扣因子 |
| 终止奖励 | $z(s)$ | 终止时刻的额外累积项 |

**回报随机变量**（与标准 RL 同构）：

$$G_t = \sum_{k=t+1}^{T} c(S_k) + z(S_T)$$

**GVF 值**：$v_\pi(s) = \mathbb{E}[G_t \mid S_t=s, A_{t:\infty}\sim\pi, T\sim\gamma]$。

### 与标准 value function 的关系

- 标准 $V^\pi$ / $Q^\pi$ 是 GVF 的特例：cumulant = 环境奖励 $r$，$\gamma$ 为常数折扣。
- **Off-policy 学习**：行为策略 $\mu$ 可与目标策略 $\pi$ 不同——Horde 用 **GTD / emphatic TD** 等从单条经验流并行更新数千 GVF。

### Span independence

[van Hasselt & Sutton (2015)](https://arxiv.org/abs/1508.04582) 指出：TD 学习 GVF 时，**每步更新成本不随预测 horizon 增长**——这是相对「belief 树展开 / 单步模型 rollout」的关键计算优势。[Modayil et al. (2014)](http://josephmodayil.com/papers/Modayil-Nexting-AdaptiveBehavior-2014.pdf) 在真实机器人上并行学习 **0.1–8 s** 多尺度 **nexting** 预测验证了这一性质。

### Horde 架构（流程）

```mermaid
flowchart LR
  sensors["传感器流 / 状态特征"]
  demons["GVF demons\n(π, c, γ, z)"]
  td["并行 off-policy TD"]
  knowledge["预测性知识库\n(数千 v̂)"]
  policy["上层策略 / options"]
  sensors --> demons
  demons --> td
  td --> knowledge
  knowledge --> policy
```

- 每个 **demon** 独立回答一个预测问题；整体系统在 **常数时间/步** 下更新（AAMAS 2011 机器人实验）。

## 工程实践

| 场景 | 做法 | 注意 |
|------|------|------|
| 辅助表征学习 | 深度 RL 中加 GVF 式辅助头（像素/特征控制等） | 需平衡主任务梯度与辅助任务数量 |
| 好奇心 / 新奇度 | 用预测误差作 intrinsic cumulant（见 White et al. 2014） | 非平稳环境下要防 collapse |
| 机器人在线监控 | Tile coding / 线性 FA + GTD 并行 nexting | 特征设计比网络深度更关键 |
| MBRL 时序抽象 | Reward-respecting subtasks (2023) 把子任务建成 GVF/options | 与短 horizon 物理模型可互补，非互斥 |

## 局限与风险

- **函数逼近误差**：大量并行 GVF 共享特征时，**干扰（interference）** 可导致部分预测长期不准。
- **问题设计**：cumulant / $\gamma$ 选不好会得到「数学上可学、任务上无用」的预测——需 **question discovery** 或与主任务耦合（见 2023 subtasks 工作）。
- **非 Alberta 范式**：当代大模型世界模型走 **潜空间想象 rollout** 路线，与 GVF **非 rollout 长期预测** 的优劣需按 horizon、随机性与样本量分情形讨论——不宜一刀切。

## 关联页面

- [Richard Sutton](../entities/richard-sutton.md) — GVF / Horde / Options 提出者
- [Bayesian Belief Analysis](./bayesian-belief-analysis.md) — belief 展开 vs GVF 直接预测的方法论对照
- [Model-Based RL](../methods/model-based-rl.md) — Sutton 一步陷阱与 GVF 替代路线
- [MDP](../formalizations/mdp.md) — GVF 建立在 MDP 交互语义之上
- [Reinforcement Learning](../methods/reinforcement-learning.md) — TD 与 off-policy 学习底座

## 参考来源

- [GVF 一手资料索引](../../sources/papers/generalized_value_functions_gvf_primary_refs.md)
- [The One-Step Trap 原始资料](../../sources/blogs/sutton_one_step_trap.md)

## 推荐继续阅读

- Sutton et al. (2011) [Horde 论文 PDF](https://www.ifaamas.org/Proceedings/aamas2011/papers/A6_R70.pdf)
- Modayil, White & Sutton (2014) [Multi-timescale Nexting](http://josephmodayil.com/papers/Modayil-Nexting-AdaptiveBehavior-2014.pdf)
- Sutton et al. (2023) [Reward-respecting Subtasks](https://arxiv.org/abs/2306.01782)
