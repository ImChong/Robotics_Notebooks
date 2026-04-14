---
type: comparison
tags: [rl, il, imitation-learning, policy-optimization, locomotion, manipulation]
status: complete
---

# RL vs 模仿学习（Imitation Learning）

RL 和 IL 是机器人策略学习的两条主干路线。两者都在学"策略 $\pi(a|s)$"，但监督信号、数据需求、能达到的行为质量完全不同。

---

## 核心对比表

| 维度 | RL（强化学习） | IL（模仿学习） |
|------|--------------|--------------|
| **监督信号** | 奖励函数 $r(s,a)$（设计） | 专家演示 $D = \{(s,a)\}$（收集） |
| **数据来源** | 自身与环境交互（无监督） | 人类/专家演示（有监督） |
| **数据效率** | 低（需大量交互） | 高（直接学专家行为） |
| **性能上限** | 理论上可超越专家 | 受专家质量上限限制 |
| **探索难度** | 高（稀疏奖励下难探索） | 无需探索（直接模仿） |
| **泛化性** | 依赖奖励函数覆盖范围 | 依赖演示数据分布 |
| **行为质量** | 取决于 reward 设计质量 | 接近专家（如演示质量高） |
| **可解释性** | 优化了什么清晰，但行为不透明 | 行为来自演示，可追溯 |

---

## 典型失败模式对比

### RL 的常见问题

**Reward Hacking**：策略利用 reward 定义的漏洞，以非预期方式得分。

> 例：locomotion reward 只惩罚摔倒，策略学会原地抖腿却不前进。

**Exploration Collapse**：在高维动作空间中无法找到有意义的 reward 信号。

> 例：精细操作任务，随机探索几乎不可能碰到"成功"状态。

**Reward Shaping 敏感性**：reward 的细微变化会导致策略行为大幅波动。

### IL 的常见问题

**Covariate Shift（分布偏移）**：测试时进入训练时未覆盖的状态，行为崩溃。

> 例：行为克隆在看过的场景里完美，一旦偏离专家轨迹就越走越偏。

**性能上界**：永远无法超越专家质量；专家演示本身有噪声时性能受限。

**演示收集成本**：高质量演示需要设备（遥操纵、动捕）和人力，代价高。

---

## 详细方法对比

### 主流 RL 方法（机器人）

| 算法 | 特点 | 典型应用 |
|------|------|---------|
| PPO | On-policy，稳定，适合仿真 | Locomotion（ANYmal, Unitree） |
| SAC | Off-policy，样本效率高，适合真实机器人 | 操作、连续控制 |
| TD3 | 确定性策略，低方差 | 精细操作 |
| AMP | 用判别器做 reward，模仿运动风格 | 自然步态 locomotion |

### 主流 IL 方法（机器人）

| 算法 | 特点 | 典型应用 |
|------|------|---------|
| BC（行为克隆） | 最简单，监督学习，covariate shift 严重 | 简单操作、初始化 |
| DAgger | 在线交互修正分布偏移 | 需要专家在线反馈 |
| ACT（Action Chunking） | 预测动作序列而非单步，减少时序误差 | Bi-manual 操作 |
| Diffusion Policy | 分布建模，多模态动作 | 精细操作、灵巧手 |
| IBC | 能量模型，隐式行为克隆 | 高维连续控制 |
| GAIL / AIRL | GAN-like，从演示中隐式学 reward | 风格模仿 |

---

## 在机器人任务上的选择指南

### 选 RL 的场景

- **任务目标清晰、reward 易定义**：locomotion（能量效率 + 速度 + 稳定性）
- **允许大量仿真交互**：可以并行仿真环境大规模训练
- **需要超越人类表现**：游戏、棋类、极限运动
- **行为空间探索需要多样性**：鲁棒性训练、域随机化

### 选 IL 的场景

- **演示容易获取，reward 难定义**：精细操作（折叠衣服、装配零件）
- **数据效率要求高**：真实机器人样本成本高，不能大量交互
- **需要快速跟上人类意图**：teleoperation、人机协作
- **行为质量要求高且一致**：医疗、服务机器人

### 两者结合（最常见）

**IL 初始化 + RL 微调**：
- 用演示数据做 BC 初始化，避免从随机策略探索
- 再用 RL 优化超越专家

**AMP（Adversarial Motion Priors）**：
- 用判别器从运动捕捉数据学习运动风格 reward
- 结合 RL 完成目标任务

**RLHF 类方法**：
- 人类偏好反馈（演示/排序）构建 reward 模型
- RL 优化 reward 模型

---

## 与其他概念的关系

```
                 ┌────────────────────────────────────┐
                 │    策略学习（Policy Learning）       │
                 └───────────┬────────────────────────┘
                    ┌────────┴──────────┐
                    ▼                  ▼
               Model-Free RL        Imitation Learning
               (PPO/SAC/TD3)      (BC/DAgger/Diffusion)
                    │                  │
                    │    融合路线       │
                    └────────┬──────────┘
                             ▼
                    AMP / RLHF / IL+RL
```

---

## 参考来源

- Sutton & Barto, *Reinforcement Learning: An Introduction* — RL 理论基础
- Pomerleau, *ALVINN: An Autonomous Land Vehicle in a Neural Network* (1989) — BC 早期工作
- Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger, 2011) — 分布偏移的经典解法
- Ho & Ermon, *Generative Adversarial Imitation Learning* (GAIL, 2016) — 从演示隐式学 reward
- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (2023) — IL 的 SOTA
- Peng et al., *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control* (2021) — RL + 演示的标志性融合

---

## 关联页面

- [Reinforcement Learning](../methods/reinforcement-learning.md) — RL 方法详细展开
- [Imitation Learning](../methods/imitation-learning.md) — IL 方法详细展开
- [Policy Optimization](../methods/policy-optimization.md) — RL 中的策略梯度方法（PPO/SAC/TD3）
- [Diffusion Policy](../methods/diffusion-policy.md) — IL 的 SOTA 方法
- [WBC vs RL](./wbc-vs-rl.md) — 另一个控制方法对比视角
- [Reward Design](../concepts/reward-design.md) — RL 选择时 reward 设计是核心挑战
- [Sim2Real](../concepts/sim2real.md) — 两种方法的 sim2real 挑战不同

## 一句话记忆

> RL 靠奖励信号自己探索，可以超越专家但 reward 难设计；IL 靠演示监督，行为质量好但受演示上界限制——实践中最强的方案往往是两者的结合。
