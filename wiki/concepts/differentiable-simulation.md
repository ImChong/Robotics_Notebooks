---
type: concept
tags: [simulation, differentiable-physics, system-identification, reinforcement-learning, gradient-based]
status: complete
updated: 2026-06-23
related:
  - ../formalizations/adjoint-sensitivity-analysis.md
  - ../entities/matrix-simulation-platform.md
  - ../entities/brax.md
  - ../entities/mujoco-mjx.md
  - ./system-identification.md
  - ../methods/model-based-rl.md
  - ./sim2real.md
sources:
  - ../../sources/courses/quadruped_control_simulation_rl_curriculum.md
summary: "Differentiable Simulation 允许对仿真 rollout 反传梯度，用于系统辨识、可微控制与样本高效策略学习；接触不连续仍是主要工程难点。"
---

# Differentiable Simulation（可微仿真）

**可微仿真**：物理引擎在 forward rollout 的同时支持 **对状态、参数或控制输入求导**，使优化与 learning 可直接利用 **解析梯度** 而非纯有限差分或黑盒 RL。

## 一句话定义

> 把仿真器变成可求导的计算图节点——**$\partial \mathcal{L}/\partial \theta$** 能穿过动力学积分，用于拟合物理参数或更新策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AD | Automatic Differentiation | 自动微分，可微仿真底层机制 |
| SysID | System Identification | 常用可微仿真做参数梯度下降 |
| RL | Reinforcement Learning | 可与 PPO 等 model-free 方法对比样本效率 |
| FoG | First-order Gradient | 一阶梯度策略优化 |
| GPU | Graphics Processing Unit | 并行 rollout + 批量反传 |
| LCP | Linear Complementarity Problem | 硬接触互补问题，求导困难 |
| MJX | MuJoCo JAX | MuJoCo 的 JAX 可微后端 |

## 为什么重要

1. **系统辨识**：课程 Ch3 用 `jax.grad` 拟合摩擦/惯量，比纯最小二乘更自然地处理 **非线性动力学闭环**。
2. **样本效率**：四足 loco 文献表明，可微仿真 + 梯度法在 **同等并行度** 下可比纯 PPO 更省样本（仍常需与高保真仿真对齐）。
3. **MPC 与 RL 的桥梁**：可微模型支持 **短视界梯度优化**，是「模型驱动」与「数据驱动」之间的中间地带。

## 核心机制

```
参数 θ（质量、摩擦、PD 增益…）
        ↓ forward integrator（可微）
状态轨迹 x(t), 损失 L
        ↓ reverse-mode AD / adjoint
∂L/∂θ  →  梯度下降更新 θ 或策略 π
```

### 接触与求导

| 接触建模 | 可微性 | 代表栈 |
|---------|--------|--------|
| 软接触 / XPBD | 较平滑，易求导 | Warp、部分 MuJoCo 近似 |
| 冲量法 | 避免 LCP 求导 | DiffTaichi 类 |
| 硬 LCP | 不连续，求导难 | 经典 MuJoCo CPU、Nimble |
| 混合双仿真器 | 简化可微模型 + 高保真对齐 | ETH 四足可微工作、MATRiX 课程叙事 |

## 在四足课程中的位置

- Ch1：定位为 **MPC（需准确模型）与 RL（黑盒探索）之间的技术桥梁**
- Ch3：可微 SysID
- Ch4–5：MATRiX 并行训练；可与 PPO 对照

## 常见误区

- **误区：「可微 = 不需要 DR」。** 模型误差与接触近似仍在，Sim2Real 仍要 DR / 补偿 / 蒸馏。
- **误区：「梯度一定比 PPO 好」。** 长视界 credit assignment 与接触不连续会让梯度噪声极大；工业界仍大量用 PPO + 并行。

## 关联页面

- 平台：[MATRiX](../entities/matrix-simulation-platform.md)、[Brax](../entities/brax.md)、[MuJoCo MJX](../entities/mujoco-mjx.md)
- 数学：[Adjoint Sensitivity Analysis](../formalizations/adjoint-sensitivity-analysis.md)
- 应用：[System Identification](./system-identification.md)、[Model-Based RL](../methods/model-based-rl.md)

## 推荐继续阅读

- Yunlong et al., *Learning Quadruped Locomotion Using Differentiable Simulation* — [arXiv:2403.14864](https://arxiv.org/abs/2403.14864)
- Freeman et al., *Brax* — [arXiv:2106.13281](https://arxiv.org/abs/2106.13281)

## 参考来源

- [sources/courses/quadruped_control_simulation_rl_curriculum.md](../../sources/courses/quadruped_control_simulation_rl_curriculum.md) — 课程 Ch1、Ch3 可微仿真与 SysID
