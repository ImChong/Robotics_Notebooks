---
title: Model-Based vs Model-Free RL 对比
type: comparison
status: complete
created: 2026-04-14
updated: 2026-04-14
summary: 从样本效率、渐近性能、工程复杂度、鲁棒性等多维度对比 Model-Based 和 Model-Free RL，给出机器人场景下的选型建议。
---

# Model-Based vs Model-Free RL 对比

## 核心定义

| 维度 | Model-Free RL | Model-Based RL |
|------|--------------|----------------|
| **世界模型** | 无，直接从交互数据学习策略/价值 | 有，先学习/使用环境动力学模型 |
| **核心操作** | 与环境交互 → 更新策略 | 交互 → 学模型 → 用模型生成虚拟数据/规划 |
| **代表算法** | PPO, SAC, TD3, DDPG | Dreamer, MBPO, PETS, TD-MPC, iLQR |

---

## 多维对比

| 对比维度 | Model-Free | Model-Based | 优胜 |
|---------|-----------|-------------|------|
| **样本效率** | 低（需大量真实交互） | 高（虚拟 rollout 补充数据） | MBRL ✅ |
| **渐近性能** | 高（无模型误差积累） | 受模型精度上限约束 | MF ✅ |
| **实现简单度** | 高（PPO 300行代码可跑） | 低（需设计模型架构、规划器） | MF ✅ |
| **真实机器人适用性** | 需要大量交互，成本高 | 样本少，适合真实硬件 | MBRL ✅ |
| **规划/预测能力** | 无（反应式） | 有（可推理未来） | MBRL ✅ |
| **分布外泛化** | 差（只记住见过的） | 模型可外推（但有误差） | 中性 |
| **调试难度** | 中（策略黑盒） | 高（模型 + 规划 + 策略三重调试） | MF ✅ |
| **计算开销** | 推理快（前向一次） | 规划需多次模型调用 | MF ✅ |

---

## 机器人场景典型对比

### 样本效率（关键指标）

在连续控制 benchmark（HalfCheetah-v3, Humanoid-v3）：

| 算法 | 类型 | 达到 5000 奖励所需步数 |
|------|------|---------------------|
| SAC | MF | ~1M steps |
| MBPO | MB | ~100K steps（10x 提升） |
| DreamerV3 | MB | ~500K steps（视任务） |
| PPO | MF | ~2M steps |

> 注：这是量级估计，实际值因任务而异

### 真实机器人

- **Model-Free（PPO/SAC）**：通常需要仿真中训练数百M步，再 sim2real 迁移，真实机器人只做微调
- **Model-Based**：可以在真实机器人上直接学习（PETS/MBPO 少量真实交互），但动力学模型学习本身有挑战

---

## 模型误差的代价

MBRL 的最大风险：**模型误差在 rollout 中累积**（compound error）

```
第 1 步误差：ε
第 k 步误差：≈ k·ε（线性累积） 或更快
```

**缓解方案：**
- 短 rollout（MBPO：1~5 步）
- 集成模型（PETS：N 个模型，取最悲观）
- 不确定性感知规划（只在高置信度区域规划）

---

## 混合方法（Model-Based + Model-Free）

实践中最优方案往往是混合：

| 方法 | 核心思想 |
|------|---------|
| **MBPO** | 短 rollout 生成虚拟数据 → SAC 更新策略 |
| **Dreamer** | 世界模型中想象轨迹 → Actor-Critic 优化 |
| **TD-MPC** | TD 价值估计 + MPPI 规划，两者互补 |
| **MPC+RL** | MPC 做高层决策，RL 策略做低层控制 |

---

## 机器人场景选型建议

```
是否有精确的动力学模型？
├── 是（MuJoCo/刚体模拟） → Model-Free（PPO/SAC）在仿真中大量训练
│   （仿真 = 免费的模型，直接 MF 效率也高）
└── 否（真实机器人/复杂接触）
    ├── 样本预算极少（< 10K 步）→ MBRL（PETS/MBPO）
    ├── 需要规划/预测能力 → TD-MPC / Dreamer
    └── 高性能要求，预算充足 → MF sim2real（PPO + 域随机化）
```

---

## 参考来源

- Janner et al., *When to Trust Your Model* (MBPO, 2019) — MBRL 样本效率分析
- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3, 2023)
- Chua et al., *PETS* (2018) — 真实机器人 MBRL 代表
- **ingest 档案：** [sources/papers/model_based_rl.md](../../sources/papers/model_based_rl.md)

---

## 关联页面

- [Model-Based RL](../methods/model-based-rl.md) — MBRL 详细介绍
- [Reinforcement Learning](../methods/reinforcement-learning.md) — RL 整体框架
- [Policy Optimization](../methods/policy-optimization.md) — Model-Free 主要算法
- [Sim2Real](../concepts/sim2real.md) — MF sim2real 是当前机器人主流路径
- [RL vs IL](./rl-vs-il.md) — 另一个重要对比维度
