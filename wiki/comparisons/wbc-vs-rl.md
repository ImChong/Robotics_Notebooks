---
type: comparison
tags: [wbc, rl, control, locomotion, humanoid]
status: complete
---

# WBC vs RL: Whole-Body Control vs Reinforcement Learning

人形机器人运动控制领域最常见的两种路线对比。

## 一句话概括

- **WBC（全身控制）**：基于模型和优化，精确、可解释、依赖精确建模
- **RL（强化学习）**：无模型或基于学习的模型，灵活、可泛化、样本效率低

---

## 核心差异

| 维度 | WBC | RL |
|------|-----|-----|
| **依赖模型** | 需要精确动力学/运动学模型 | 不需要（model-free），或学一个近似模型 |
| **样本效率** | 高（一次优化） | 低（需要大量交互） |
| **泛化能力** | 受模型精度限制 | 可泛化到新环境 |
| **计算实时性** | QP/NMPC 可实时 | 需要 GPU 或大量前期计算 |
| **处理接触切换** | 需要精心设计状态机 | 自然处理（端到端） |
| **对不确定性鲁棒性** | 差（模型误差敏感） | 较强（通过随机化等） |
| **人工介入程度** | 高（手动设计约束、任务权重） | 低（reward 设计相对简单） |
| **理论保证** | 有稳定性/收敛性保证 | 弱 |
| **行为自然度** | 依赖参考轨迹质量 | 可通过 AMP 等方法学习自然运动 |
| **硬件要求** | CPU 可实时（QP 求解快） | 推理需要 GPU（高维神经网络） |

---

## 各自适合的场景

### WBC 更适合
- 已知精确模型的场景（工业机器人、结构化环境）
- 需要精确轨迹跟踪的任务（焊接、装配）
- 需要硬约束（安全边界、关节限位严格执行）
- 算力受限的嵌入式部署
- 需要可解释性、可调试性

### RL 更适合
- 模型难以获得的场景（软体、复杂接触、非结构化地形）
- 需要泛化到新任务/新环境（跨地形、跨任务）
- 任务难以手工设计控制策略（灵巧操作、跑酷）
- 有充足仿真资源的场景（Isaac Lab 大规模并行）
- 接受策略不可解释

---

## 主流融合架构（关键！）

纯 WBC 或纯 RL 在复杂任务上都有短板，当前最强的人形控制系统几乎都是融合架构。

### 架构 1：RL 高层 + WBC 低层（主流）

```
High-Level RL Policy
  └─ 输出：质心速度参考 / 落脚点 / 接触时序
         ↓
Low-Level WBC (TSID/HQP)
  └─ 输出：关节力矩
```

**代表**：ETH ANYmal 系列、Boston Dynamics Atlas（推测）

**优点**：
- RL 处理高层决策和地形适应
- WBC 保证底层动力学一致性和安全约束
- 两层解耦，各自可独立调试

### 架构 2：RL 端到端（直接输出力矩）

```
RL Policy（PPO/SAC）
  └─ 输入：本体感知（IMU/关节）
  └─ 输出：关节位置目标 or 力矩
```

**代表**：Unitree legged_gym、ETH Learning to Walk in Minutes、OpenAI Dactyl

**优点**：
- 简单直接，无需手工设计中间表示
- 接触切换自然处理
- 可以端到端优化

**缺点**：
- 可解释性差
- 需要大量仿真训练
- 真实部署 sim2real gap 较大

### 架构 3：WBC 生成演示 + IL/RL 蒸馏

```
WBC/MPC 在线求解（teacher）
  └─ 生成高质量演示轨迹
         ↓
IL/RL（student）学习压缩策略
  └─ 推理时不需要 WBC
```

**代表**：DART（通过 MPC 生成数据训练神经网络策略）

**优点**：
- 利用 WBC 的精确性生成训练数据
- 推理时只需轻量神经网络，无 QP 求解开销
- 保留了 WBC 的行为质量

### 架构 4：ASE/CALM/AMP — LLC + HLC 分层

```
High-Level Controller（HLC）
  └─ 输出：任务指令（如"向右走 0.5m/s"）
         ↓
Low-Level Controller（LLC，预训练 RL）
  └─ 接收 HLC 指令，执行物理上可行的动作
```

**代表**：
- **AMP（Adversarial Motion Priors）**：判别器从 MoCap 数据学习运动风格，作为 reward
- **ASE（Adversarial Skill Embeddings）**：LLC 学习多样化技能嵌入，HLC 组合技能
- **CALM**：类似 ASE，用条件化潜变量控制风格

**优点**：
- LLC 预训练一次，HLC 可以快速适配新任务
- 自然产生人类化运动

### 架构 5：MPC-WBC 集成（经典人形架构）

```
MPC（Centroidal Dynamics）
  └─ 规划质心轨迹 + 接触力分配
         ↓
WBC（TSID/HQP）
  └─ 分解为关节加速度 + 力矩
```

见：[MPC 与 WBC 集成](../concepts/mpc-wbc-integration.md)

---

## 技术栈路线建议

### 入门路线
1. 先学 WBC 理论基础（TSID、HQP、Centroidal Dynamics）
2. 在 Pinocchio + Crocoddyl 上实现简单的 WBC
3. 再学 RL（PPO on MuJoCo locomotion 任务）
4. 最后研究融合架构

### 工程选型参考

| 场景 | 推荐方案 |
|------|---------|
| 已知地形 + 精确任务 | MPC-WBC |
| 复杂地形 locomotion | RL 端到端 + 域随机化 |
| 自然运动风格 | AMP + RL |
| 操作 + 行走 | RL HLC + WBC LLC |
| 快速原型（周级） | legged_gym + PPO |
| 追求最高性能 | 融合架构（RL HLC + WBC LLC） |

---

## 参考来源

- Peng et al., *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control* (2021) — RL 与运动风格融合路线代表
- Peng et al., *ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters* (2022) — LLC+HLC 分层 RL
- Sentis & Khatib, *Synthesis of Whole-Body Behaviors through Contact and Collision Avoidance* — WBC 理论基础
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (2021) — RL + Adaptation 的 sim2real
- [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md) — TSID/HQP/Crocoddyl ingest 摘要
- [WBC vs RL 论文导航](../../references/papers/whole-body-control.md)

---

## 关联页面

- [Whole-Body Control](../concepts/whole-body-control.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [Sim2Real](../concepts/sim2real.md)
- [MPC 与 WBC 集成](../concepts/mpc-wbc-integration.md)（融合架构的核心实现）
- [TSID](../concepts/tsid.md)（WBC 最典型的执行层实现）
- [RL vs IL](./rl-vs-il.md)（另一角度的控制策略对比）

## 一句话记忆

> WBC 是用精确模型解优化问题，RL 是从交互数据学策略——两者的融合架构（RL 高层 + WBC 低层）代表了当前人形机器人控制的工程最优解。
