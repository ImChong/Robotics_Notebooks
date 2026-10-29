---
title: Model-Based vs Model-Free RL 对比
type: comparison
status: complete
created: 2026-04-14
updated: 2026-10-29
summary: 从模型使用方式、样本效率、模型偏差、工程复杂度与机器人部署约束对比 Model-Based 和 Model-Free RL；强调结论依赖任务、模型质量和算力预算。
---

# Model-Based vs Model-Free RL 对比

## 核心定义

| 维度 | Model-Free RL | Model-Based RL |
|------|--------------|----------------|
| **动力学模型** | 不显式使用动力学模型生成 rollout 或规划（仍可使用观测历史等结构） | 显式学习或使用已知动力学模型，进行预测、规划或模型 rollout |
| **核心操作** | 与环境交互 → 更新策略 | 交互 → 学模型 → 用模型生成虚拟数据/规划 |
| **代表算法** | PPO, SAC, TD3, DDPG | Dreamer, MBPO, PETS, TD-MPC, iLQR |

---

## 多维对比

| 对比维度 | Model-Free | Model-Based | 优胜 |
|---------|-----------|-------------|------|
| **样本效率** | 通常依赖较多交互；可借助仿真、并行采样和演示缓解 | 可用模型复用数据，但学习模型和控制的误差会影响收益 | 依任务与模型而定 |
| **最终性能** | 不受显式模型误差限制，但仍受数据、函数逼近和优化影响 | 受模型偏差影响，也可能因规划或样本效率获益 | 无普遍胜者 |
| **实现复杂度** | 基础策略梯度路径通常较直接 | 通常还需设计/辨识模型、规划或模型 rollout | 取决于算法与工程栈 |
| **真实机器人适用性** | 常在仿真训练后迁移；直接真机采样成本高 | 可提高数据复用，但模型偏差和安全探索仍需处理 | 取决于数据与验证条件 |
| **规划/预测能力** | 无（反应式） | 有（可推理未来） | MBRL ✅ |
| **分布外泛化** | 取决于训练覆盖、结构与迁移方法 | 模型可能外推，也可能在分布外快速失准 | 都不能保证 |
| **调试难度** | 中（策略黑盒） | 高（模型 + 规划 + 策略三重调试） | MF ✅ |
| **计算开销** | 推理快（前向一次） | 规划需多次模型调用 | MF ✅ |

---

## 阅读边界

- Model-Free / Model-Based 是训练与决策机制的区分，不等于“无预测 / 有预测”；策略可隐式利用时序信息，而 Model-Based 方法也可能把规划与学习策略组合起来。
- 样本效率、渐近性能、泛化与计算开销都依赖任务、模型误差、数据预算和实现。表格中的“优胜”是常见倾向，不是算法类别的保证。
- 下方 HalfCheetah / Humanoid 的步数为粗略量级，奖励阈值、环境版本、实现、随机种子与硬件未完整对齐；不应据此当作严格的同基准排名。需要定量选型时，应补齐对应实验来源和协议。

## 机器人场景典型对比

### 样本效率（关键指标）

连续控制中的样本效率高度依赖环境、奖励、并行度、实现和调参预算。不同论文的步数数字通常不能直接横向比较，因此不把某个固定步数或倍率当作通用结论。

- **MBPO**：用短模型 rollout 补充真实经验，改善样本效率；收益取决于模型误差与任务。
- **DreamerV3**：在潜空间想象中优化策略；其跨任务结果不能直接换算成上述统一奖励门槛。
- **SAC / PPO**：直接从环境交互学习；与模型方法比较时须统一环境、交互预算、评测种子和指标。

### 真实机器人

- **Model-Free（PPO/SAC）**：通常需要仿真中训练数百M步，再 sim2real 迁移，真实机器人只做微调
- **Model-Based**：可以利用有限真实数据学习模型并在模型内规划/生成 rollout；是否能安全、高效地直接真机学习，取决于建模误差、探索约束和系统安全机制。

---

## 模型误差的代价

MBRL 的最大风险：**模型误差在 rollout 中累积**（compound error）

```
第 1 步误差：ε
第 k 步误差：≈ k·ε（线性累积） 或更快
```

**缓解方案：**
- 短 rollout（MBPO：1~5 步）
- 集成模型（如 PETS）估计预测不确定性，并在规划中降低高不确定轨迹的风险
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

## 复核说明（2026-10-12）

原表将“样本效率高”“真机适用性强”等写成固定胜负，容易把特定论文结果误作算法类别保证。本次复核将结论改为依赖任务、模型误差、仿真能力、算力和安全预算；固定步数示例替换为可比性说明。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| SAC | Soft Actor-Critic | 连续控制常用的 off-policy 最大熵算法 |
| Dreamer | Dreamer (World Models) | 在潜空间想象中训练的 MBRL 代表 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| iLQR | iterative Linear Quadratic Regulator | 对非线性系统迭代线性化求解的轨迹优化方法 |
| MBRL | Model-Based Reinforcement Learning | 先学/用环境模型再规划或想象 rollout 的强化学习 |
| DreamerV3 | Dreamer version 3 | 在潜空间想象中训练、单一超参跨 150+ 任务通用的世界模型智能体 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| IL | Imitation Learning | 从专家演示学习策略，奖励难定义时的主路线 |

## 参考来源

- Janner et al., *When to Trust Your Model* (MBPO, 2019) — MBRL 样本效率分析
- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3, 2023)
- Chua et al., *PETS* (2018) — 真实机器人 MBRL 代表
- **ingest 档案：** [sources/papers/model_based_rl.md](../../sources/papers/model_based_rl.md)

---

## 关联页面

- [Model-Based RL](../methods/model-based-rl.md) — MBRL 详细介绍
- [MBPO](../methods/mbpo.md) — 短模型 rollout + SAC 的混合方法代表
- [Reinforcement Learning](../methods/reinforcement-learning.md) — RL 整体框架
- [Policy Optimization](../methods/policy-optimization.md) — Model-Free 主要算法
- [Sim2Real](../concepts/sim2real.md) — MF sim2real 是当前机器人主流路径
- [RL vs IL](./rl-vs-il.md) — 另一个重要对比维度
