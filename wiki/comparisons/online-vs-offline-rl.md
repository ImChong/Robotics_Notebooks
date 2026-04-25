---
type: comparison
tags: [rl, offline-rl, online-rl, data-efficiency, distribution-shift, locomotion]
status: complete
related:
  - ../methods/reinforcement-learning.md
  - ../methods/imitation-learning.md
  - ../concepts/sim2real.md
  - ../comparisons/rl-vs-il.md
sources:
  - ../../sources/papers/locomotion_rl.md
  - ../../sources/papers/policy_optimization.md
summary: "Online RL vs Offline RL"
updated: 2026-04-25
---

# Online RL vs Offline RL

Online RL 和 Offline RL 是两种根本不同的学习范式。两者都在优化同一个目标（累积奖励），但对**数据来源**的要求截然不同，导致适用场景和瓶颈完全不同。

## 核心对比

| 维度 | Online RL | Offline RL |
|------|----------|-----------|
| **数据来源** | 策略与环境实时交互生成 | 固定数据集（人类演示 / 旧策略轨迹） |
| **探索** | ✅ 主动探索（exploration） | ❌ 无法探索，只能从已有数据学 |
| **数据效率** | 低（需大量交互） | 高（充分利用已有数据） |
| **分布偏移** | 无（训练分布 = 运行分布） | ⚠️ 严重（OOD 动作无真实反馈） |
| **仿真依赖** | 高（通常需要仿真器） | 低（可用历史数据训练） |
| **性能上限** | 理论上无上限（能超越数据集） | 受数据集质量上限限制 |
| **安全性** | 低（探索会产生危险动作） | 高（不与真实环境交互） |
| **代表算法** | PPO、SAC、TD3 | CQL、IQL、TD3+BC、Decision Transformer |

## Online RL

### 工作原理

```
策略 π → 执行动作 → 环境反馈 r,s' → 更新 π → 循环
```

策略自己探索环境，收集数据，不断自我改进。

### 优势

- **可超越数据集**：没有数据质量上限，策略可以发现数据集中没有的好行为
- **无分布偏移问题**：数据是当前策略生成的，和训练分布一致
- **理论收敛性**：在合适条件下可收敛到最优策略

### 劣势

- **仿真器依赖**：真实机器人做 online RL 成本高（磨损、安全风险），通常需要仿真
- **样本效率低**：人形机器人 locomotion 通常需要 1-10 亿步仿真
- **探索危险**：早期随机动作可能损坏机器人（真实环境）

### 机器人场景适用性

✅ **最适合**：仿真中训练 locomotion 策略（Isaac Lab、legged\_gym），Sim2Real 后部署

❌ **不适合**：直接在真实机器人上做 online RL（成本高、安全风险）

## Offline RL

### 工作原理

```
固定数据集 D = {(s,a,r,s')} → 从数据中学策略 → 部署
```

关键挑战：**分布偏移（distributional shift）**——学到的策略可能访问数据集中没有的状态-动作对，没有真实反馈无法纠正。

### 分布偏移问题

```
数据集 D: {(s₁,a₁), (s₂,a₂), ...}  ← 旧策略或人类生成

新策略 π 可能选择: (s_new, a_ood)  ← Out-Of-Distribution
                        ↑
              没有真实 Q 值，bootstrap 误差无法校正
```

解决方案：**保守性约束**——惩罚 OOD 动作的 Q 值估计。

### 代表方法

| 方法 | 保守性机制 |
|------|---------|
| **CQL**（Conservative Q-Learning） | 对 OOD 动作的 Q 值加惩罚项 |
| **IQL**（Implicit Q-Learning） | 不显式评估 OOD 动作，用 expectile regression |
| **TD3+BC** | 行为克隆正则项约束策略不偏离数据集 |
| **Decision Transformer** | 条件 sequence model，直接生成动作序列 |

### 机器人场景适用性

✅ **适合**：
- 有大量历史遥操作数据（manipulation / 灵巧手）
- 策略 fine-tuning（先 offline 预训练，再 online 微调）
- 安全关键场景（医疗、精密装配）

❌ **不适合**：
- 需要超越数据集质量的任务
- 数据质量差（旧策略生成的低质量轨迹）
- 高动态 locomotion（分布偏移严重）

## Offline → Online 混合策略

实践中越来越常见的方案：

```
1. Offline 预训练（大量历史数据 → 合理初始策略）
2. Online 微调（少量真实/仿真交互 → 超越数据集上限）
```

代表工作：IQL + online fine-tuning、Cal-QL。

优点：
- 利用已有数据，减少从零探索的危险
- 最终性能不受数据集上限限制

## 在人形机器人中的选择依据

| 场景 | 推荐范式 | 原因 |
|------|---------|------|
| 仿真训练 locomotion | Online RL（PPO） | 仿真成本低，需要探索 |
| 真实机器人 fine-tuning | Offline RL + Online | 保守起步，有限探索 |
| Manipulation（遥操作数据） | Offline RL | 高质量演示数据充足 |
| 新任务 / 新机体 | Online RL（先仿真） | 没有先验数据 |
| 低数据预算 | Offline RL | 数据复用效率高 |

## 参考来源

- Levine et al., *Offline Reinforcement Learning: Tutorial, Review, and Perspectives* (2020) — Offline RL 综述
- Kumar et al., *Conservative Q-Learning for Offline Reinforcement Learning* (2020) — CQL 原始论文
- Kostrikov et al., *Offline Reinforcement Learning with Implicit Q-Learning* (2021) — IQL

## 关联页面

- [Reinforcement Learning](../methods/reinforcement-learning.md) — Online RL 的主流算法（PPO、SAC）
- [Imitation Learning](../methods/imitation-learning.md) — Offline RL 与 BC 的边界：IL 不需奖励，Offline RL 需要奖励标注
- [Sim2Real](../concepts/sim2real.md) — Online RL 依赖仿真；Offline RL 可缓解仿真依赖
- [RL vs IL](./rl-vs-il.md) — 三角关系：Online RL / Offline RL / IL 的数据与监督信号对比
