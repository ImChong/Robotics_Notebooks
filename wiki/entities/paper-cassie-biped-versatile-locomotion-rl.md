---
type: entity
tags: [biped, cassie, reinforcement-learning, sim2real, pd-control, domain-randomization]
status: stable
summary: "Cassie 双足：双历史 I/O 架构 + 任务随机化，统一多种动态技能；文中给出策略/PD 分频与 PD 增益缩放随机化等工程细节。"
updated: 2026-05-12
related:
  - ../queries/legged-humanoid-rl-pd-gain-setting.md
  - ../queries/domain-randomization-guide.md
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/rl_pd_action_interface_locomotion.md
---

# Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control（Cassie）

**一句话定义**：在 Cassie 上，用 **长/短双历史** 的观测–动作序列输入统一表达周期与非周期运动，再配合 **任务层随机化**，在仿真中学会多技能并 **直接 sim2real** 到硬件。

## 为什么重要

- 把「**策略多慢、PD 多快、增益要不要随机化**」写成可对照的 **公开工程细节**：文中报告 **策略约 33 Hz**、**关节 PD 内环约 2 kHz**；训练中对名义 PD 增益做 **约 0.7–1.3 倍缩放随机化**。
- 与仓库内 [Domain Randomization 指南](../queries/domain-randomization-guide.md) 的刚度缩放示例可 **并排阅读**：一文偏 **通用 checklist**，一文偏 **双足 Cassie 实测区间**。

## 核心机制（提炼）

- **Dual-history**：长 horizon 捕获步态周期与地形缓变；短 horizon 强化触地瞬态与快扰动响应。
- **技能统一**：行走、跑步、跳跃等共享同一策略族，通过 **命令与随机化** 切换行为先验。

```mermaid
flowchart TB
  subgraph obs["观测历史"]
    L["长历史分支"]
    S["短历史分支"]
  end
  pol["策略网络<br/>约 33 Hz"]
  tgt["关节目标<br/>q_des 或等价接口"]
  pd["PD 内环<br/>约 2 kHz"]
  L --> pol
  S --> pol
  pol --> tgt --> pd
  dr["训练期 PD 增益<br/>0.7 to 1.3 x 名义"]
  dr -.-> pd
```

## 与 Kp / Kd 设置的关系

- **分频优先**：先锁「策略一步内 PD 子步数 / 真机控制循环」，再调名义 `Kp/Kd`；否则 DR 区间会对不上真实带宽。
- **随机化是结构化假设**：0.7–1.3 是 **缩放名义表** 的窄区间实践，与「±30% 标称」类宽随机化并不矛盾，服务于 **不同保守程度** 的 sim2real。

## 参考来源

- [RL+PD 动作接口与增益设计论文索引](../../sources/papers/rl_pd_action_interface_locomotion.md)
- *Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control*, [arXiv:2401.16889](https://arxiv.org/abs/2401.16889)

## 关联页面

- [Legged / Humanoid RL 中 Kp/Kd 设置](../queries/legged-humanoid-rl-pd-gain-setting.md)
- [Domain Randomization 指南](../queries/domain-randomization-guide.md)
- [Sim2Real](../concepts/sim2real.md)
- [Cassie 迭代式 sim2real  locomotion](./paper-cassie-iterative-locomotion-sim2real.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2401.16889.pdf)
