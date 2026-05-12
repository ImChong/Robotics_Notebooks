---
type: entity
tags: [humanoid, reinforcement-learning, sim2real, legged, digit, pd-control]
status: stable
summary: "UC Berkeley：全尺寸 Digit 人形上大规模并行 RL + 因果 Transformer 策略，零样本户外行走与 sim2real 流水线（含关节 PD 部署链）。"
updated: 2026-05-12
related:
  - ../queries/legged-humanoid-rl-pd-gain-setting.md
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../entities/legged-gym.md
sources:
  - ../../sources/papers/rl_pd_action_interface_locomotion.md
---

# Real-World Humanoid Locomotion with Reinforcement Learning（Digit）

**一句话定义**：在 Agility Digit 全尺寸人形上，用 **大规模并行仿真 + 域随机化** 训练 **因果 Transformer**，从本体感觉与动作历史自回归预测下一步关节指令，经 **低层 PD/跟踪** 上真机，实现零样本户外行走与扰动恢复。

## 为什么重要

- 把「**高维人形 + 盲走 + 户外材质变化**」收进一条可公开的 **sim→厂商仿真→硬件** 流水线，便于和经典 MPC / 厂商控制器对照。
- 对 **Kp/Kd 话题** 的价值在于：动作仍是 **离散策略输出 → 连续关节阻抗跟踪**；论文给出 **逐关节增益表**（随关节而异），适合作为 **人形 PD 量级** 的文献锚点，而不是孤立抄一组数。

## 核心机制（提炼）

- **两阶段学习**：先 **全观测教师策略** \(\pi_s(a|s)\)，再训 **仅本体学生** \(\pi_o(a|o_{:t},a_{:t-1})\)，学生损失混合 **模仿 + POMDP 强化**。
- **序列模型**：因果 Transformer 将 \((o,a)\) 历史 token 化，用自注意力做 **上下文内适应**（不更新权重）。
- **仿真**：Isaac Gym 千级并行环境 + 地形/动力学随机化；闭链与 Digit 机构在文中有专门仿真处理叙述。

```mermaid
flowchart LR
  subgraph sim["仿真训练"]
    E["并行随机环境<br/>Isaac Gym"]
    T["教师 pi_s"]
    S["学生 Transformer pi_o"]
    E --> T
    T --> S
  end
  subgraph gate["闸门与部署"]
    M["厂商高保真仿真"]
    H["真机 Digit"]
    PD["关节 PD 跟踪<br/>增益表见论文"]
  end
  S --> M --> H
  H --> PD
```

## 与 Kp / Kd 设置的关系

- 精读时应以 **论文附录 / 补充材料中的 PD 表** 为准；公开讨论里常以 **髋部量级 \(K_p\approx 200\) N·m/rad、\(K_d\approx 10\) N·m·s/rad** 作为可读锚点，**左右肢与各关节仍有差异**。
- 调参时把 **策略时间步、仿真子步、PD 更新率** 与表放在同一页系统图里核对，避免只改 `stiffness` 不改分频。

## 参考来源

- [RL+PD 动作接口与增益设计论文索引](../../sources/papers/rl_pd_action_interface_locomotion.md)（本条在索引中的摘录与链接）
- Radosavovic et al., *Real-World Humanoid Locomotion with Reinforcement Learning*, [arXiv:2303.03381](https://arxiv.org/abs/2303.03381)

## 关联页面

- [Legged / Humanoid RL 中 Kp/Kd 设置](../queries/legged-humanoid-rl-pd-gain-setting.md)
- [Sim2Real](../concepts/sim2real.md)
- [Locomotion](../tasks/locomotion.md)
- [legged_gym](./legged-gym.md)

## 推荐继续阅读

- [项目页 Learning Humanoid Locomotion](https://learning-humanoid-locomotion.github.io/)
