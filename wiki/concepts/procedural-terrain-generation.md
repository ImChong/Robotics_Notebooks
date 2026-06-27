---
type: concept
tags: [terrain, simulation, domain-randomization, reinforcement-learning, curriculum-learning]
status: complete
updated: 2026-06-23
related:
  - ./domain-randomization.md
  - ./curriculum-learning.md
  - ./terrain-adaptation.md
  - ../entities/legged-gym.md
  - ../entities/matrix-simulation-platform.md
  - ../entities/extreme-parkour.md
  - ../entities/quadruped-control-curriculum.md
sources:
  - ../../sources/courses/quadruped_control_simulation_rl_curriculum.md
summary: "Procedural Terrain Generation 在仿真中程序化生成坡、碎石、台阶与随机凸起，配合 DR 与课程学习训练四足越障泛化策略。"
---

# Procedural Terrain Generation（程序化地形生成）

**程序化地形生成**：在仿真器中 **按规则或噪声参数** 自动生成多样地形 mesh/高度场，而非手工建模单个场景，用于 **批量训练** 四足越障与泛化评估。

## 一句话定义

> 用算法「造地形」—— 每回合随机坡度、台阶高度、碎石分布，逼策略学 **地形不变的运动原理** 而非记地图。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DR | Domain Randomization | 地形参数常纳入随机化 |
| RL | Reinforcement Learning | 并行环境每 reset 换新地形 |
| HF | Height Field | 高度场，常见地形表示 |
| CL | Curriculum Learning | 由易到难逐步加大地形难度 |
| Sim2Real | Simulation to Real | 地形多样性支撑户外迁移 |
| PPO | Proximal Policy Optimization | 课程默认训练算法 |
| UE5 | Unreal Engine 5 | MATRiX 侧视觉地形渲染 |

## 为什么重要

课程 Ch5 与 Project 2 要求：

- 生成 **坡、碎石、台阶、随机凸起** 等 ≥3 类地形
- 对比 **有/无 DR** 策略泛化
- 与 **五阶段课程学习** 联用：先平地 Trot，再逐步加难度

## 常见生成方式

| 类型 | 参数示例 | 训练目的 |
|------|---------|---------|
| 斜坡 | 倾角、摩擦 | 上坡加速、下坡制动 |
| 台阶 | 高度、宽度、随机序列 | 抬腿高度与落足 |
| 碎石/噪声高度场 | 振幅、相关长度 | 接触不确定性 |
| 离散障碍 | 方块尺寸分布 | 避障与步态调整 |

实现参考：[Legged Gym](../entities/legged-gym.md) `terrain.py`、MATRiX JSON 场景、Isaac Lab terrain importer。

## 与 DR、课程学习的关系

```
程序化地形 → 提供随机化「载体」
     ↓
DR 采样物理参数（摩擦、质量…）
     ↓
课程学习调度难度分布
     ↓
PPO 并行训练
```

## 常见误区

- **地形无限难 + 无课程**：策略学不到基础步态，样本浪费。
- **只视觉难、物理简单**：Sim2Real 仍失败；碰撞体须与视觉对齐（MATRiX 强调 UE–MuJoCo 一致）。

## 关联页面

- [仿真物理保真度链路选型指南](../queries/simulation-physics-fidelity.md) — 本页所述物理/仿真要素在保真度链路（建模 ① → 数值 ② → 接触 ③ → 随机化 ④）中的定位
- [Domain Randomization](./domain-randomization.md)
- [Curriculum Learning](./curriculum-learning.md)
- [Terrain Adaptation](./terrain-adaptation.md)
- [Quadruped Control Curriculum](../entities/quadruped-control-curriculum.md)
- [Extreme Parkour](../entities/extreme-parkour.md)

## 推荐继续阅读

- Rudin et al., *Learning to Walk in Minutes* — 地形课程经典
- [Domain Randomization Guide](../queries/domain-randomization-guide.md)

## 参考来源

- [sources/courses/quadruped_control_simulation_rl_curriculum.md](../../sources/courses/quadruped_control_simulation_rl_curriculum.md) — 课程 Ch5 与 Project 2
