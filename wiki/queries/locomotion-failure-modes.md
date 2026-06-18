---
type: query
tags: [rl, locomotion, debugging, training, simulation]
status: complete
updated: 2026-06-18
related:
  - ./robot-policy-debug-playbook.md
  - ./reward-shaping-guide.md
  - ../concepts/sim2real.md
  - ../entities/legged-gym.md
sources:
  - ../../sources/papers/policy_optimization.md
summary: "RL 训练失败模式分析：原地踏步、关节抖动（高频振荡）、双脚起跳与策略崩溃等 locomotion 典型现象的诊断与奖励/PD 对策。"
---

# Locomotion RL 训练失败模式分析

**RL 训练**时若出现 **原地踏步**、**关节抖动** 或步态异常，可用本页按症状快速对照奖励项、PD 与课程设计。

> **Query 产物**：本页由以下问题触发：「locomotion RL 训练时常见的失败模式有哪些？怎么诊断？」
> 综合来源：[Reward Shaping Guide](./reward-shaping-guide.md)、[Robot Policy Debug Playbook](./robot-policy-debug-playbook.md)、[Legged Gym](../entities/legged-gym.md)

---

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Reward | Reward Function | 塑造强化学习策略行为的标量反馈 |
| PD | Proportional–Derivative | 关节位置/阻抗底层控制，策略输出常为其 setpoint |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| DR | Domain Randomization | 训练时随机化仿真参数以提升跨域鲁棒迁移 |

## 典型失败现象与诊断

### 1. 原地踏步 (The "Stationary Trot")
- **症状**：机器人能够维持平衡并有明显的踏步动作，但拒绝向前移动，或者仅在原地微幅抖动。
- **原因**：
  - **前进奖励太低**：前进带来的正奖励无法抵消前进过程中产生的关节功耗惩罚或碰撞惩罚。
  - **惩罚项过重**：例如 `action_rate` 或 `energy` 惩罚太大，导致机器人认为“不动”是最优解。
- **对策**：增加 `lin_vel_xy` 权重，或引入 **Curriculum**（先训平衡，再加初速度命令）。

### 2. 双脚起跳 (The "Bunny Hop")
- **症状**：机器人不交替迈步，而是像兔子一样双脚（或四足同时）跳跃前进。
- **原因**：
  - **Air Time 奖励滥用**：给摆动脚在空中的时间给奖励，但没有限制单次只能有一只脚在空中。
  - **缺少步态约束**：奖励函数没有诱导交替性（例如惩罚双脚同时离地）。
- **对策**：限制 `feet_air_time` 的最大值，或者加入对角步态（Trot）的显式引导。

### 3. 关节高频抖动 (The "High-frequency Jitter")
- **症状**：步态整体正确，但观察到执行器在高频振荡，甚至在仿真中发出异响。
- **原因**：
  - **缺少平滑惩罚**：没有对 `action_rate`（动作变化率）或关节加速度进行惩罚。
  - **控制频率与 PD 参数不匹配**：仿真频率太低而 PD 增益过高。
- **对策**：增加 `action_rate` 惩罚权重；减小 PD 增益；检查仿真步长。

### 4. 躯干下蹲/内八字 (The "Squatting/Pigeon-toed")
- **症状**：为了不跌倒，机器人学会了极低重心的蹲行，或者足尖向内扣。
- **原因**：
  - **过度追求稳定性**：蹲得越低越难倒，机器人通过“钻漏洞”来最大化不跌倒奖励。
  - **缺少基准姿态约束**：没有对默认关节角度 $q_{default}$ 的偏离进行惩罚。
- **对策**：加入 `base_height` 奖励（维持特定高度）；加入 `joint_regularization` 惩罚（偏离默认姿态）。

### 5. 策略崩溃 (Policy Collapse)
- **症状**：训练过程中 Reward 突然断崖式下跌，且无法恢复。
- **原因**：
  - **学习率过大**：导致 PPO 更新跳出了信任区域。
  - **环境随机性太大**：域随机化（DR）范围在初期开得太广，模型无法收敛。
- **对策**：减小学习率；使用学习率衰减；逐步开启域随机化范围。

---

## 诊断工具 Checklist

- [ ] **可视化检查**：在仿真器中慢速观察失败瞬间，看是哪一个关节先失控。
- [ ] **奖励分量分析**：在 TensorBoard 中单独绘制每项 Reward 的曲线。如果某项惩罚（Negative Reward）数值远大于正奖励，机器人必然会停止运动。
- [ ] **动作分布检查**：检查策略输出的均值和标准差。如果均值始终在 Limit 边界上，说明控制量已饱和。

---

## 关联页面
- [机器人策略排障手册](./robot-policy-debug-playbook.md)
- [奖励函数设计指南](./reward-shaping-guide.md)
- [Sim2Real 概念](../concepts/sim2real.md)
- [Legged Gym 实体](../entities/legged-gym.md)

## 参考来源
- [sources/papers/policy_optimization.md](../../sources/papers/policy_optimization.md)
- Rudin et al., *Learning to Walk in Minutes* (2022).
