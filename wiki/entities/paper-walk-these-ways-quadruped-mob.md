---

type: entity
tags: [quadruped, reinforcement-learning, sim2real, generalization, legged-gym, mit]
status: stable
summary: "四足 MoB：单一策略嵌入多种步态与风格参数，部署时人类可调参以适配分布外地形与任务；开源 Walk These Ways 控制器。"
updated: 2026-07-19
arxiv: "2212.03238"
related:
  - ../entities/legged-gym.md
  - ../entities/paper-anymal-walk-minutes-parallel-drl.md
  - ../queries/legged-humanoid-rl-pd-gain-setting.md
  - ../tasks/locomotion.md
  - ../tasks/stair-obstacle-perceptive-locomotion.md
sources:
  - ../../sources/papers/rl_pd_action_interface_locomotion.md
---

# Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior

**一句话定义**：学习 **单一条件策略** \(\pi(a|c,b)\)：在 **同一平坦训练分布** 上，用少量 **行为参数 \(b\)** 切换步态族（频率、摆腿高度、躯干姿态等），从而在 **未见地形与扰动** 上通过 **在线调参** 快速找到可行解，而不是立刻重训。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| PD | Proportional–Derivative | 关节位置/阻抗底层控制，策略输出常为其 setpoint |
| OOD | Out-of-Distribution | 分布外样本/未见场景，泛化评测关注点 |
| Kp | Proportional Gain | PD 控制的位置误差增益，影响刚度与响应 |
| Kd | Derivative Gain | PD 控制的速度误差增益，抑制振荡 |
| legged_gym | Legged Gym | 足式机器人 RL 训练的常用开源框架 |
| DR | Domain Randomization | 训练时随机化仿真参数以提升跨域鲁棒迁移 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

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

## 实验与评测

- 量化指标、消融与 sim2real / 实机结果见 **原文 PDF** 与 [参考来源](#参考来源)；本页正文侧重方法结构与知识库交叉引用。

## 与其他工作对比

- 与同期 **baseline、PD 内环、纯模仿或纯 RL** 等路线的差异见原文实验章节；知识库内相关概念页见 **关联页面**。
- 与 [Learning to Adapt（Nature MI 2025）](./paper-learning-to-adapt-bio-inspired-quadruped-gait.md) 对照：MoB 用 **连续行为参数 b** 索引多样解；后者用 **离散 gait ID（Γ\*）+ BGS 参考 + πG 生物力学指标** 做 **8 步态在线切换** 与 **辅助步态恢复**。

## 参考来源

- [RL+PD 动作接口与增益设计论文索引](../../sources/papers/rl_pd_action_interface_locomotion.md)
- Margolis & Agrawal, *Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior*, [arXiv:2212.03238](https://arxiv.org/abs/2212.03238)

## 关联页面

- [Legged / Humanoid RL 中 Kp/Kd 设置](../queries/legged-humanoid-rl-pd-gain-setting.md)
- [Learning to Walk in Minutes（并行 DRL）](./paper-anymal-walk-minutes-parallel-drl.md)
- [legged_gym](./legged-gym.md)
- [Locomotion](../tasks/locomotion.md)
- [Learning to Adapt（Nature MI 2025 四足多步态）](./paper-learning-to-adapt-bio-inspired-quadruped-gait.md)

## 推荐继续阅读

- [项目页 Walk These Ways](https://gmargo11.github.io/walk-these-ways/)
