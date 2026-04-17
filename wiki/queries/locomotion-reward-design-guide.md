---
title: Locomotion 奖励函数设计指南
type: query
status: complete
created: 2026-04-14
updated: 2026-04-14
summary: 系统整理 RL 训练足式/人形机器人 locomotion 的奖励函数设计原则、常用奖励项分类、调参策略和常见失败模式。
sources:
  - ../../sources/papers/reward_design.md
  - ../../sources/papers/locomotion_rl.md
---

> **Query 产物**：本页由以下问题触发：「怎么设计 locomotion RL 的奖励函数？」
> 综合来源：[Reinforcement Learning](../methods/reinforcement-learning.md)、[Locomotion](../tasks/locomotion.md)、[Sim2Real](../concepts/sim2real.md)、[Reward Design](../concepts/reward-design.md)

# Locomotion 奖励函数设计指南

## 奖励函数的核心哲学

好的 locomotion 奖励设计满足：
1. **任务奖励**：奖励"达成目标"（速度跟踪、到达目的地）
2. **风格约束**：惩罚"不想要的行为"（能量消耗、关节震荡、不稳定姿态）
3. **稀疏 vs 稠密**：locomotion 通常用稠密奖励（每步都有反馈）

---

## 标准奖励项分类

### A. 任务奖励（Task Reward）
核心奖励，权重最高

| 奖励项 | 公式 | 典型权重 |
|--------|------|---------|
| 线速度跟踪 | $\exp(-\|v_{xy} - v_{cmd}\|^2 / \sigma)$ | 1.0 |
| 角速度跟踪（yaw） | $\exp(-\|w_z - w_{cmd}\|^2 / \sigma)$ | 0.5 |
| 目标方向行走 | $v \cdot \hat{d}$（速度在目标方向上的投影） | 1.0 |

**注意：** 用 `exp(-x²/σ)` 而不是线性 `-|x|`，在目标附近梯度更平滑，训练更稳定。

### B. 稳定性奖励（Stability Reward）
惩罚会导致跌倒或不自然姿态的行为

| 奖励项 | 公式 | 典型权重 |
|--------|------|---------|
| 躯干高度保持 | $-\|h - h_{ref}\|^2$ | 0.2 |
| 躯干姿态 | $-(\phi^2 + \theta^2)$（roll, pitch） | 0.2 |
| 基座线速度 Z 轴 | $-(v_z)^2$ | 0.1 |
| 基座角速度 X/Y 轴 | $-(w_{xy})^2$ | 0.1 |

### C. 能效奖励（Efficiency Reward）
鼓励省力、平滑的动作

| 奖励项 | 公式 | 典型权重 |
|--------|------|---------|
| 关节力矩惩罚 | $-\|\tau\|^2$ | 0.0001 |
| 关节速度惩罚 | $-\|\dot{q}\|^2$ | 0.0001 |
| 动作变化惩罚（平滑性） | $-\|\Delta a\|^2 = -\|a_t - a_{t-1}\|^2$ | 0.005 |
| 电能消耗 | $-|\tau \cdot \dot{q}|$ | 0.0001 |

### D. 接触奖励（Contact Reward）
鼓励合理的步态和接触模式

| 奖励项 | 说明 | 典型权重 |
|--------|------|---------|
| 步态对称性 | 左右足接触相位差应接近 0.5 | 0.1 |
| 摆动脚高度 | 摆动相时足端离地 > 阈值 | 0.1 |
| 双支撑时长 | 限制双支撑比例（行走） | 0.05 |
| 接触冲击 | 惩罚大的接触力变化率 | 0.1 |

### E. 安全约束（Safety Reward）
防止关节过载和硬件损坏

| 奖励项 | 公式 | 典型权重 |
|--------|------|---------|
| 关节限制接近 | $-\text{clip}(q - q_{limit}, 0, \infty)^2$ | 10.0（高惩罚） |
| 关节速度超限 | 类似 | 5.0 |
| 躯干碰地 | 碰到则 -100（终止奖励） | -100 |

---

## 奖励权重调参策略

### 1. 分阶段调参

```
阶段 1：只开任务奖励，调到能行走
阶段 2：加稳定性奖励，调到步态稳
阶段 3：加能效惩罚，调到自然省力
阶段 4：全部开启，微调权重
```

### 2. 权重量级参考

所有奖励项的总和，在一个 timestep 内大致在 0 ~ 5 之间。
- 任务奖励：1.0（归一化基准）
- 稳定性：0.1 ~ 0.5 倍任务奖励
- 能效：0.001 ~ 0.01 倍（太大会让机器人不敢动）
- 安全惩罚：10 ~ 100 倍（要让策略"害怕"）

### 3. 使用 `exp` 而非线性

```python
# 推荐（连续、在零点有梯度）
r_vel = torch.exp(-((v_xy - v_cmd)**2).sum(dim=-1) / 0.25)

# 不推荐（在目标处梯度为零）
r_vel = -(v_xy - v_cmd).norm(dim=-1)
```

---

## 常见失败模式与修复

| 失败现象 | 原因 | 修复方法 |
|---------|------|---------|
| 机器人原地转圈（spinning） | 角速度奖励权重不足 | 增加 yaw 跟踪奖励 |
| 机器人爬行而不站立 | 躯干高度奖励缺失 | 增加 `r_base_height` |
| 关节剧烈抖动 | 动作平滑惩罚太小 | 增加 `r_action_rate` |
| 步幅过大/不自然 | 能量惩罚不足 | 增加 `r_torques` |
| 速度达到但姿态崩溃 | 稳定性奖励权重不够 | 平衡任务与稳定性权重 |
| 跑步而不行走 | 无步态约束 | 加步频/摆动相奖励 |
| 训练后期性能下降 | 奖励 scale 随课程变化 | 对奖励做 normalize |

---

## legged_gym 标准奖励结构（参考）

legged_gym 使用 `cfg.rewards.scales` 字典：

```python
class rewards:
    class scales:
        # 任务
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        # 稳定性
        torques = -0.0002
        dof_vel = -0.0
        dof_acc = -2.5e-7
        action_rate = -0.01
        # 姿态
        base_height = -0.0
        feet_air_time = 1.0
        collision = -1.0
        # 安全
        dof_pos_limits = -10.0
```

---

## 奖励工程中的 Sim2Real 注意点

- **力矩惩罚尺度**：仿真 vs 真实机器人力矩量级差异需归一化
- **接触奖励**：仿真接触完美，真实有噪声 → 接触奖励在真实中不可靠
- **平滑性奖励**：真实机器人需要更强的平滑惩罚（噪声放大问题）
- **能量惩罚**：过强的能量惩罚可能导致策略在真实机器人上过于保守

---

## 参考来源

- Rudin et al., *Learning to Walk in Minutes* (2022) — legged_gym 奖励设计参考
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (2021) — PPO + 奖励设计
- Margolis et al., *Walk These Ways: Tuning Robot Walking to Suit All Humans* (2023) — 多步态奖励设计

---

## 关联页面

- [Reward Design](../concepts/reward-design.md) — 奖励函数设计的通用原则
- [Locomotion](../tasks/locomotion.md) — 奖励设计的应用场景
- [Reinforcement Learning](../methods/reinforcement-learning.md) — RL 训练框架
- [Curriculum Learning](../concepts/curriculum-learning.md) — 奖励与课程经常配合使用
- [Sim2Real](../concepts/sim2real.md) — 奖励设计影响 sim2real 迁移效果
- [legged_gym](../entities/legged-gym.md) — legged_gym 的具体奖励实现
