---
type: query
tags: [reward, rl, locomotion, training, curriculum]
status: complete
summary: "Reward Design 实战指南"
updated: 2026-04-25
sources:
  - ../../sources/papers/reward_design.md
---

# Reward Design 实战指南

> **Query 产物**：本页由以下问题触发：「从零设计 locomotion RL 的 reward 函数，有哪些核心原则和常见陷阱？」
> 综合来源：[reward-design](../concepts/reward-design.md)、[curriculum-learning](../concepts/curriculum-learning.md)、[locomotion](../tasks/locomotion.md)、[rl-algorithm-selection](./rl-algorithm-selection.md)

## TL;DR 决策路径

```
任务是否有清晰的成功标准？
├─ 是 → 考虑稀疏 reward（+1/0），配合 Curriculum 渐进难度
│        代表：OpenAI RAPID，EUREKA
└─ 否 → 使用密集 reward，手动分解子目标
         代表：legged_gym（速度跟踪 + 稳定性 + 能耗）

子目标是否可以独立度量？
├─ 是 → 加权求和（独立权重，消融调参）
└─ 否 → 层级 reward 或 AMP（对抗模仿 prior）

reward 是否导致不期望行为（reward hacking）？
├─ 是 → 加惩罚项，或用 potential-based shaping
└─ 否 → 保持简单
```

## 核心设计原则

### 1. Locomotion Reward 基本结构（legged_gym 范式）

```python
# legged_gym 典型 reward 组合
reward = (
    w_vel   * track_lin_vel()      # 线速度跟踪（主要目标）
  + w_ang   * track_ang_vel()      # 角速度跟踪
  + w_base  * base_height()        # 基座高度稳定
  - w_cont  * contact_forces()     # 接触力惩罚（避免冲击）
  - w_jvel  * joint_vel()          # 关节速度惩罚（平滑运动）
  - w_jacc  * joint_acc()          # 关节加速度惩罚（减少抖动）
  - w_act   * action_rate()        # 动作变化率惩罚（连续性）
  - w_torq  * torques()            # 力矩惩罚（节能）
)
```

典型权重比例（参考 legged_gym）：
| 分项 | 权重范围 | 备注 |
|------|---------|------|
| 线速度跟踪 | 1.0–2.0 | 主要目标，最高权重 |
| 关节速度惩罚 | 0.01–0.05 | 防止过激动作 |
| 动作变化率 | 0.01–0.1 | 平滑性 |
| 接触力 | 0.001–0.01 | 避免硬冲击 |

### 2. Potential-Based Reward Shaping（Ng 1999）

理论保证：任何形如 $r' = r + \gamma\Phi(s') - \Phi(s)$ 的变换不改变最优策略。

实践应用：
- $\Phi(s)$ = 负距离目标（越近越好）
- $\Phi(s)$ = 当前速度与目标速度之差的负值

**陷阱**：非 potential-based 的额外 reward（如终止奖励）会改变最优策略。

### 3. 参数化 Reward（Walk These Ways 范式）

将步态参数作为命令向量，reward 跟踪参数目标：

```python
def gait_reward(obs, cmd):
    # cmd = [lin_vel_x, lin_vel_y, ang_vel_z, step_freq, step_height, ...]
    vel_error = |obs["base_vel"] - cmd["target_vel"]|
    freq_error = |obs["contact_freq"] - cmd["step_freq"]|
    return -vel_error - 0.5 * freq_error
```

优势：单策略支持多步态，无需分别训练。

### 4. AMP 对抗模仿（Peng 2021）

当步态质量难以用手工 reward 描述时，用示范数据定义 reward：

$$r_{AMP} = \log D_\phi(s, a, s')$$

其中 $D_\phi$ 是判别器（能否分辨真实示范 vs 策略动作）。

适用场景：
- 需要自然步态（非机械感）
- 有高质量 motion capture 数据
- 手工 reward 导致明显不自然行为

## 常见陷阱与修复

| 陷阱 | 症状 | 修复 |
|------|------|------|
| **Reward Hacking** | 策略发现意外高分行为（如滑行/静止） | 加多维度惩罚 + 检查所有 reward 极值情况 |
| **权重不平衡** | 某个分项主导，其他分项训练无效 | 归一化各分项范围至 [-1, 1]，再加权 |
| **稀疏 reward 收敛慢** | 早期奖励几乎为 0，策略无法起步 | 加 Curriculum（从简单任务开始）或 dense shaping |
| **惩罚项过强** | 策略学会"不动"来规避惩罚 | 降低惩罚权重 / 加运动激励 reward |
| **关节角度超限** | 关节打到限位，电机过热 | 加关节角度 penalty，或用 soft limit 约束 |
| **Episode 终止奖励泄露** | 策略学会触发终止（如故意摔倒）来获得 bonus | 消除终止奖励，或改为 shaping 形式 |

## Curriculum 设计原则

1. **从成功率触发晋级**：统计最近 N 个 episode 的成功率，超过阈值（如 80%）才提升难度
2. **渐进地形**：平地 → 小坡 → 台阶 → 楼梯 → 随机地形
3. **速度课程**：低速（0.5 m/s）→ 中速（1.0 m/s）→ 高速（2.0+）
4. **避免太难的初始条件**：初始难度过高 = 零梯度 = 策略不收敛

## 参考来源

- [sources/papers/reward_design.md](../../sources/papers/reward_design.md) — Rudin legged_gym / Walk These Ways / EUREKA / Portelas 综述
- [Locomotion Reward Design Guide](./locomotion-reward-design-guide.md) — 更偏 locomotion 任务的详细实践

## 关联页面

- [Reward Design](../concepts/reward-design.md) — 奖励函数设计理论
- [Curriculum Learning](../concepts/curriculum-learning.md) — 课程学习与 reward 的配合
- [Locomotion](../tasks/locomotion.md) — locomotion 任务整体
- [RL Algorithm Selection](./rl-algorithm-selection.md) — 算法选型也影响 reward 设计

## 一句话记忆

> Reward = 主目标跟踪（高权重）+ 稳定性约束（中权重）+ 能耗/平滑惩罚（低权重）；Curriculum = 按成功率自动晋级难度。
