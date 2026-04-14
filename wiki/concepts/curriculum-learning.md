---
title: Curriculum Learning（课程学习）
type: concept
status: complete
created: 2026-04-14
updated: 2026-04-14
summary: 从简单到复杂的渐进式训练策略，在机器人 RL 中用于解决稀疏奖励、地形多样性和任务复杂度梯度问题。
---

# Curriculum Learning（课程学习）

## 是什么

Curriculum Learning 是一种训练策略：在学习早期提供更简单的任务或环境，随着策略能力提升逐渐增加难度，模拟人类"从简单到复杂"的学习过程。

核心思想：**合理安排训练样本/环境难度的顺序，比随机采样更高效**。

---

## 为什么重要

在机器人强化学习中，直接从最终任务开始训练常常遇到：
- **稀疏奖励陷阱**：代理从未到达目标区域，奖励信号全零，无从学习
- **地形多样性**：复杂地形（台阶 / 斜坡 / 石堆）需先在平地上掌握基础步态
- **接触丰富的操作**：物体抓取 → 放置 → 装配，每步都依赖前一步

---

## 典型实现形式

### 1. 手动课程（Manual Curriculum）
人工预定义难度序列：
```
Stage 1: 平地行走（100 steps）
Stage 2: 轻微随机地形（±0.05m）
Stage 3: 中等地形（±0.15m, 斜坡 10°）
Stage 4: 复杂地形（台阶 0.2m, 石堆）
```

**优点**：可控、可解释  
**缺点**：需要领域专家，不自适应

### 2. 自动课程（Automatic / Adaptive Curriculum）
根据策略当前表现动态调整难度：
- **成功率阈值**：当前 stage 成功率 > 80% → 提升难度
- **ALP-GMM**（Absolute Learning Progress）：追踪每个难度区域的学习进度，主动采样最有提升空间的区域

### 3. 地形课程（Terrain Curriculum for Locomotion）
legged_gym / Isaac Lab 的标准做法：
```python
# 成功率 > 0.8 → 地形难度 +1
# 成功率 < 0.5 → 地形难度 -1
terrain_level = clip(terrain_level + delta, 0, max_level)
```
8192 并行环境，每个 env 独立地形等级，GPU 上并行更新。

### 4. 任务课程（Goal Curriculum）
- **HER + 课程**：从简单目标开始，扩展目标分布
- **POET**（Paired Open-Ended Trailblazer）：生成-解决配对，双进化过程

---

## 在机器人中的典型应用

| 场景 | 课程策略 | 代表工作 |
|------|---------|---------|
| 四足/双足 locomotion | 地形难度分级 | legged_gym (Rudin 2022) |
| Dexterous manipulation | 物体位置随机化范围扩大 | OpenAI Dactyl (2019) |
| Humanoid 站立/行走 | 初始姿态随机化幅度 + 地形 | Agility Robotics, HUMA |
| Sim2Real 迁移 | 域随机化参数范围逐步扩大 | ETH ANYmal 系列 |
| 多技能学习 | 技能难度拓扑排序 | ASE, CALM |

---

## 与域随机化的关系

域随机化（Domain Randomization）和课程学习经常配合使用：
- **域随机化**：训练集覆盖参数分布的范围（解决 sim2real 差距）
- **课程学习**：控制参数分布的 **采样顺序**（解决训练效率）

典型组合：
```
初始：小随机化范围（e.g., mass ±5%）
中期：随策略成熟，逐步扩大（mass ±30%）
最终：全随机化范围，确保 sim2real 泛化
```

---

## 工程注意事项

- **课程切换时机**：过早升级 → 策略尚未收敛，出现遗忘；过晚 → 浪费训练时间
- **双向调整**：难了降级，容了升级，避免卡在某一阶段
- **并行环境不同步**：8192 个 env 各有自己的难度等级，避免全局同步导致的振荡
- **奖励归一化**：不同难度的奖励量级不同，需在课程变化时注意归一化

---

## 参考来源

- Bengio et al., *Curriculum Learning* (ICML 2009) — 课程学习原始论文，证明合理样本顺序加速收敛
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning* (2022) — legged_gym 地形课程实现
- Portelas et al., *Automatic Curriculum Learning for Deep RL: A Short Survey* (2020) — 自动课程学习综述

---

## 关联页面

- [Reinforcement Learning](../methods/reinforcement-learning.md) — 课程学习是 RL 的训练策略，不改变算法本身
- [Sim2Real](./sim2real.md) — 域随机化课程化是 sim2real 的关键手段
- [Locomotion](../tasks/locomotion.md) — 地形课程在 locomotion 训练中普遍使用
- [Reward Design](./reward-design.md) — 课程与奖励稀疏性密切相关，配合 reward shaping 使用
- [legged_gym](../entities/legged-gym.md) — legged_gym 内置地形课程实现
- [Privileged Training](./privileged-training.md) — teacher-student 框架常与课程学习结合
