---
type: comparison
tags: [mpc, rl, control, locomotion, comparison, engineering-selection]
status: stable
summary: "MPC vs RL：控制策略选型对比"
updated: 2026-06-10
sources:
  - ../../sources/papers/mpc.md
  - ../../sources/papers/policy_optimization.md
  - ../../sources/papers/mpc_rl_arxiv_2606_05687.md
---

# MPC vs RL：控制策略选型对比

**背景**：MPC（模型预测控制）和 RL（强化学习）是当前机器人运动控制领域的两大主流范式。MPC 基于显式动力学模型在线求解最优控制，RL 离线学习隐式策略。两者在假设、计算模式和适用场景上都有本质差异。

## 一句话定义

> MPC 是"实时思考，有模型有规划"；RL 是"离线练好，推理时直接执行"——前者依赖精确模型但可在线适应，后者不需模型但泛化受训练分布约束。

---

## 核心维度对比

| 维度 | MPC | RL |
|------|-----|----|
| **模型依赖** | 需要精确动力学模型 | 不需要（Model-Free RL）或弱依赖（Model-Based RL） |
| **计算时机** | 在线求解 OCP（每控制周期） | 离线训练，推理时仅前向传播 |
| **推理速度** | 受求解器限制（毫秒~几十毫秒） | 极快（神经网络前向，< 1ms） |
| **样本效率** | 高（利用模型，少量数据） | 低（Model-Free 需百万~亿级交互） |
| **可解释性** | 高（优化目标和约束显式） | 低（策略为黑盒） |
| **约束处理** | 显式，硬约束（关节限位/力约束） | 隐式（奖励惩罚，不保证满足） |
| **泛化能力** | 弱（模型误差大时降级） | 强（域随机化后对扰动更鲁棒） |
| **真机 gap** | 依赖 SysID 精度 | sim2real gap，但可通过域随机化缓解 |
| **开发周期** | 长（建模+调参+求解器集成） | 中等（奖励设计+训练时间） |
| **高动态场景** | 需要快速求解器，否则延迟成问题 | 天然实时（推理 < 1ms） |

---

## 什么时候选 MPC

**适合 MPC 的场景**：

1. **需要硬约束保证**：安全临界场景（关节力矩上限、稳定性约束）
2. **任务目标频繁变化**：在线更改速度命令 / 步态 / 轨迹，MPC 在线重规划
3. **模型准确且可辨识**：系统参数已知，动力学误差 < 5%
4. **需要可解释决策**：工业/医疗场景，需要知道控制器在做什么
5. **低速精细操作**：操作臂精密操作，速度要求不高但精度要求高

**典型 MPC 工作流**：
```
系统辨识 → 建立动力学模型 → 设计代价函数 + 约束 → 选择求解器
→ 实时 OCP 求解（50-200Hz）→ 执行第一步控制
```

**代表系统**：ETH RSL OCS2（ANYmal）、MIT Cheetah3 MPC、Unitree 官方 MPC 控制器

---

## 什么时候选 RL

**适合 RL 的场景**：

1. **高动态/高速运动**：奔跑、跳跃、翻滚——MPC 求解器跟不上动态变化
2. **模型难以精确建模**：软体机器人、复杂接触、变形地面
3. **需要鲁棒性而非精确性**：崎岖地形、外力扰动、传感器噪声
4. **端到端感知→动作**：视觉输入直接→关节命令，MPC 难以处理高维感知
5. **策略复用 / finetune**：在多个硬件上快速迁移，修改奖励而非重建模型

**典型 RL 工作流**：
```
仿真环境 → 奖励函数设计 → PPO/SAC 训练 → 域随机化
→ Teacher-Student 蒸馏 → 真机部署（推理 < 1ms）
```

**代表系统**：legged_gym + Isaac Lab、parkour policy（ETH/CMU）、Unitree RL 策略

---

## 混合架构：MPC + RL

实际高性能系统往往结合两者优势，常见有三条轴：

### A. 部署期分层（RL 上 / MPC 下 或相反）

```
高层 RL 策略（步态/步位决策）
         ↓
低层 MPC 执行器（实时接触力优化）
```

- **RL 上层**：学习步态切换、行为风格、长时序规划
- **MPC 下层**：保证物理约束，精确接触力分配

代表工作：
- CMU Humanoid Locomotion（2023）：RL policy + MPC 接触规划
- ETH RSL：OCS2 MPC + RL 初始化
- MIT Cheetah：MPC + RL 结合的高速奔跑

### B. 训练期 MPC 指导、部署期纯 RL

```
训练：CD-MPC 批求解 → 预测地标奖励 → PPO
部署：仅 MLP 策略（无在线 MPC）
```

- **MPC 仅训练时**：质心 MPC 轨迹转为 landmark guidance reward，把长时域物理结构写进奖励
- **RL 部署时**：推理极快，无求解器延迟；适合大规模并行仿真内嵌 MPC

代表工作：
- [MPC-RL](../entities/paper-mpc-rl-humanoid-locomotion-manipulation.md)（Caltech/JHU, 2026, arXiv:2606.05687）：[πⁿ MPC](../methods/pi-mpc.md) 支撑 4096 env 长时域 CD-MPC；Themis 真机 loco-manipulation（含 290 kg 推车）

### C. 残差 / 在线 MPC 增强

- **残差 RL on MPC**：测试时 MPC 仍在环，RL 学残差修正
- **MPC-over-RL**（如 Sumo）：高层采样 MPC 在 RL WBC 命令空间规划

---

## 决策矩阵

```
你的主要需求是什么？
│
├── 保证约束满足（安全关键）→ MPC
├── 高速/动态运动（> 2m/s 奔跑/跳跃）→ RL
├── 视觉输入 → RL
├── 模型已知且精确 → MPC
├── 需要鲁棒扰动恢复 → RL（或 RL+MPC 混合）
├── 快速原型（周内出策略）→ RL（legged_gym 10分钟出策略）
├── 精密操作（工业机械臂）→ MPC
└── 未知/复杂接触（爬行/操作软物体）→ RL
```

---

## 与 model-based-vs-model-free 对比页的区别

[model-based-vs-model-free](./model-based-vs-model-free.md) 讨论的是 RL 内部的 Model-Based vs Model-Free 之分。  
本页讨论的是**控制范式**层面：基于优化的控制（MPC）vs 基于学习的控制（RL），是更高层的工程选型问题。

---

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| OCP | Optimal Control Problem | MPC 每步求解的有限时域最优控制问题 |
| SysID | System Identification | 系统辨识，估计物理/动力学参数 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| ANYmal | ANYbotics Quadruped | ANYbotics 的四足机器人研究平台 |
| legged_gym | Legged Gym | 足式机器人 RL 训练的常用开源框架 |
| Isaac Lab | NVIDIA Isaac Lab | 基于 Omniverse 的机器人学习训练框架 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| RMA | Rapid Motor Adaptation | 从历史轨迹隐式估计环境参数的快速运动自适应 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |

## 参考来源

- [sources/papers/mpc.md](../../sources/papers/mpc.md) — MPC 核心论文（Di Carlo 2018, Sleiman 2021）
- [sources/papers/policy_optimization.md](../../sources/papers/policy_optimization.md) — RL 策略优化论文（PPO, Rudin 2022）
- Kumar et al., *RMA: Rapid Motor Adaptation* (2021) — RL sim2real 真机控制
- Di Carlo et al., *Dynamic Locomotion in the MIT Cheetah 3* (2018) — MPC 足式控制

---

## 关联页面

- [Model Predictive Control](../methods/model-predictive-control.md) — MPC 详细介绍
- [Reinforcement Learning](../methods/reinforcement-learning.md) — RL 详细介绍
- [WBC vs RL](./wbc-vs-rl.md) — 相关的控制架构对比
- [Model-Based vs Model-Free](./model-based-vs-model-free.md) — RL 内部的方法对比
- [MPC Solver Selection](../queries/mpc-solver-selection.md) — MPC 求解器选型指南
- [MPC-RL](../entities/paper-mpc-rl-humanoid-locomotion-manipulation.md) — 训练期 MPC 地标奖励 + 部署期纯 RL
- [π MPC](../methods/pi-mpc.md) — parallel-in-horizon ADMM 求解器（MPC-RL 批训练后端）

---

## 推荐继续阅读

- Bledt & Kim, *Implementing Regularized Predictive Control for High-Performance Quadrupedal Locomotion*
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep RL*

## 一句话记忆

> MPC 有模型会算约束，适合精密可控场景；RL 练好了推理极快，适合高速鲁棒场景；顶级系统往往两者结合——RL 管策略，MPC 管约束。
