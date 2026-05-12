---
type: entity
tags: [quadruped, reinforcement-learning, sim2real, generalization, legged-gym]
status: stable
summary: "四足 MoB：单一策略嵌入多种步态与风格参数，部署时人类可调参以适配分布外地形与任务；开源 Walk These Ways 控制器。"
updated: 2026-05-12
related:
  - ../entities/legged-gym.md
  - ../entities/paper-anymal-walk-minutes-parallel-drl.md
  - ../queries/legged-humanoid-rl-pd-gain-setting.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rl_pd_action_interface_locomotion.md
---

# Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior

**一句话定义**：学习 **单一条件策略** \(\pi(a|c,b)\)：在 **同一平坦训练分布** 上，用少量 **行为参数 \(b\)** 切换步态族（频率、摆腿高度、躯干姿态等），从而在 **未见地形与扰动** 上通过 **在线调参** 快速找到可行解，而不是立刻重训。

## 为什么重要

- 解决「训练奖励只能偏置一种辅助行为、换任务就崩」的痛点：把 **多样性** 预埋在 **结构化行为空间** 里。
- 与 **固定 PD + 增益随机化** 路线天然衔接：低层仍是经典力矩接口，上层用 \(b\) 解释 **接触与身体形态** 的变化。

## 核心机制（提炼）

- **MoB（Multiplicity of Behavior）**：训练时鼓励在 **同一任务奖励** 下存在 **多个等价最优** 的解，由 \(b\) 索引。
- **部署**：人类或高层模块调节 \(b\)，在楼梯、滑地、侧推等 OOD 场景间 **快速试错**。

```mermaid
flowchart LR
  train["仅平地训练"]
  pi["策略 pi a mid c b"]
  train --> pi
  dep["部署: 调 b"]
  pi --> dep
  ood["OOD 地形与扰动"]
  dep --> ood
```

## 与 Kp / Kd 设置的关系

- 当你已能跑通 **legged_gym 式默认 PD**，下一步常不是盲目加大 DR，而是像本文一样 **先获得一族可切换的低层行为**，再决定增益随机化区间。
- **文献号**：本文 arXiv 为 **[2212.03238](https://arxiv.org/abs/2212.03238)**（不是「Walk in Minutes」）。

## 参考来源

- [RL+PD 动作接口与增益设计论文索引](../../sources/papers/rl_pd_action_interface_locomotion.md)
- Margolis & Agrawal, *Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior*, [arXiv:2212.03238](https://arxiv.org/abs/2212.03238)

## 关联页面

- [Legged / Humanoid RL 中 Kp/Kd 设置](../queries/legged-humanoid-rl-pd-gain-setting.md)
- [Learning to Walk in Minutes（并行 DRL）](./paper-anymal-walk-minutes-parallel-drl.md)
- [legged_gym](./legged-gym.md)
- [Locomotion](../tasks/locomotion.md)

## 推荐继续阅读

- [项目页 Walk These Ways](https://gmargo11.github.io/walk-these-ways/)
