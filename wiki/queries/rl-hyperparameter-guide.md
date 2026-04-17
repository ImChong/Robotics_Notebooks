---
type: query
tags: [rl, ppo, sac, hyperparameter, locomotion, training]
related:
  - ../methods/policy-optimization.md
  - ../concepts/reward-design.md
  - ../concepts/curriculum-learning.md
  - ../methods/model-predictive-control.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/policy_optimization.md
  - ../../sources/papers/reward_design.md
---

# RL 超参数调节指南（locomotion 专用）

> **Query 产物**：本页由问题「locomotion RL 训练超参数如何选择和调节？」触发。
> Query 类型：操作指南
> 生成日期：2026-04-15
> 问题：训练腿式机器人 locomotion 策略时，PPO/SAC 的关键超参数如何调节？

---

## 背景

locomotion RL 的超参数调节与标准游戏 RL 有显著差异：仿真并行度高、observation/action 空间维度大、reward 稀疏性问题突出、sim2real gap 要求训练出鲁棒的策略。

---

## PPO 超参数 Checklist

### 核心超参数

| 参数 | locomotion 推荐范围 | 说明 |
|------|------------------|------|
| `num_envs` | 2048–8192 | 并行仿真数量，越多越稳定；受显存限制 |
| `num_steps` | 24–64 | 每次 rollout 的步数（太短 → 高方差，太长 → 过时数据） |
| `batch_size` | `num_envs × num_steps / num_minibatches` | 典型 4096–16384 |
| `num_minibatches` | 4–8 | 每批 rollout 的 minibatch 数量 |
| `update_epochs` | 4–10 | 每批数据的训练轮次 |
| `clip_range` (ε) | 0.1–0.2 | PPO clip；locomotion 通常用 0.2，精细任务用 0.1 |
| `value_loss_coef` | 0.5–2.0 | value function 损失权重 |
| `entropy_coef` | 0.001–0.01 | 探索熵正则化；locomotion 中不宜过大 |
| `gamma` (折扣因子) | 0.97–0.99 | locomotion 长程任务用 0.99 |
| `lambda` (GAE) | 0.90–0.97 | bias-variance 折中；0.95 是默认起点 |
| `lr` (学习率) | 1e-4–3e-4 | Adam 优化器；可用线性 decay |
| `max_grad_norm` | 0.5–1.0 | 梯度裁剪，稳定训练 |

### 调参优先级

```
1. num_envs × num_steps → 数据吞吐量（最影响训练速度）
2. gamma + lambda → reward 传播（最影响收敛方向）
3. clip_range + update_epochs → 更新稳定性
4. entropy_coef → 探索 vs 利用
```

---

## SAC 超参数 Checklist

| 参数 | locomotion 推荐范围 | 说明 |
|------|------------------|------|
| `replay_buffer_size` | 1e6–5e6 | 样本重用率高，需要大 buffer |
| `batch_size` | 256–4096 | SAC 对 batch_size 不敏感，可大 |
| `lr_actor/critic` | 3e-4–1e-3 | 独立学习率；critic 可稍大 |
| `tau` (软更新) | 0.005–0.01 | target network 更新速率 |
| `alpha` (温度) | 自动调节 | 使用自动 entropy tuning（SAC paper 推荐） |
| `target_entropy` | `-dim(action)` | 通常取 `-action_dim`，即每维约 -1 nats |
| `update_frequency` | 1–4 | 每步环境交互后的梯度更新次数 |

---

## Reward 函数调参 Checklist

### 常见 reward 项权重范围（locomotion）

| Reward 项 | 推荐权重 | 说明 |
|----------|---------|------|
| 前进速度奖励 | 1.0（归一化基准） | 任务核心信号 |
| 姿态稳定惩罚 | -0.1 ~ -0.5 | 防止躯干倾斜 |
| 能量/扭矩惩罚 | -0.01 ~ -0.1 | 减少能耗，泛化更好 |
| 关节速度惩罚 | -0.001 ~ -0.01 | 防止抖动，sim2real 关键 |
| 接触力惩罚 | -0.01 ~ -0.1 | 防止硬接触；改善地形适应 |
| 足端滑移惩罚 | -0.1 ~ -1.0 | 防止打滑；sim2real 必须项 |
| 存活奖励 | 0.1 ~ 1.0 | 防止策略"坐下" |

### 调参规则

1. **先调任务奖励，再调正则惩罚**：确认策略能完成任务后再加惩罚项
2. **惩罚项权重从小开始**：初始设 0.01–0.05，观察是否抑制主任务学习
3. **能量惩罚 = sim2real 利器**：减少关节抖动和扭矩尖峰，是最廉价的 sim2real 改善手段
4. **课程学习优先于奖励 shaping**：难以 shape 的任务改用渐进式课程

---

## 常见问题诊断

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 策略不动（reward ≈ 0） | entropy 太低 / reward 太稀疏 | 提高 entropy_coef / 添加进度奖励 |
| 训练不稳定（reward 剧烈振荡） | lr 过大 / clip_range 过大 | 降低 lr / 降至 0.1 / 增加 max_grad_norm |
| 策略在真机上抖动 | 关节速度 / 扭矩惩罚不足 | 提高关节速度惩罚权重 |
| 策略速度快但步态奇怪 | 缺少步态对称性约束 | 添加脚频约束 / air time reward |
| 训练后期无提升（平台期） | 数据探索不足 | 增加 entropy / 降低课程难度 |

---

## 关联页面
- [策略优化方法](../methods/policy-optimization.md)
- [Reward 设计](../concepts/reward-design.md)
- [课程学习](../concepts/curriculum-learning.md)
- [Locomotion 任务](../tasks/locomotion.md)

## 参考来源
- [policy_optimization.md](../../sources/papers/policy_optimization.md)
- [reward_design.md](../../sources/papers/reward_design.md)
