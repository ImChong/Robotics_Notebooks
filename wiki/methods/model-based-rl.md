---
type: method
tags: [rl, model-based, planning, locomotion, sample-efficiency]
status: complete
---

# Model-Based RL（基于模型的强化学习）

**Model-Based RL（MBRL）**：在强化学习中，智能体显式学习或利用环境的动力学模型，通过在模型中规划或生成虚拟经验来提升样本效率。

## 一句话定义

> 先学会世界是怎么运作的（模型），再用这个模型来练技能——而不是只靠和真实环境反复试错。

---

## 为什么重要

Model-Free RL 的核心问题：**样本效率低**。

在机器人任务中：
- 真实机器人每次交互有物理成本（时间、硬件损耗）
- 仿真虽然可以并行加速，但高保真度仿真依然慢
- 复杂操作任务需要大量探索

MBRL 的价值：
- **样本效率**：利用模型生成虚拟经验（Model Rollouts），减少真实交互
- **规划能力**：有了模型可以做前向搜索/轨迹优化
- **迁移性**：模型可以跨任务复用（学一次世界模型，解多个任务）

---

## 核心范式

### 范式 1：Dyna 架构（经典）

$$\text{真实经验} \rightarrow \text{学习模型} \rightarrow \text{模型生成虚拟经验} \rightarrow \text{更新策略}$$

- 与真实环境交互收集少量数据
- 学习动力学模型 $\hat{f}(s, a) \rightarrow s'$
- 用模型采样大量虚拟轨迹
- 用真实 + 虚拟经验更新 value function / 策略

### 范式 2：基于规划的 MBRL

直接在模型中做轨迹优化，不显式学习策略。

代表：MPC（Model Predictive Control）、MPPI、CEM。

$$a^* = \arg\max_{\{a_t\}_{t=0}^{H}} \sum_{t=0}^{H} r(s_t, a_t)$$

每步都重新规划，不需要预先学好的策略。

### 范式 3：世界模型（World Model）

在潜空间中学习紧凑的世界表示，在潜空间中规划和想象。

代表：Dreamer 系列。

$$s_t \sim q_\phi(s_t | s_{t-1}, a_{t-1}, o_t), \quad \hat{o}_t \sim p_\theta(\hat{o}_t | s_t)$$

---

## 代表性算法

### Dreamer / DreamerV3（Hafner et al.）

**核心思想**：学习一个紧凑的循环世界模型（RSSM），在潜空间中想象未来，用想象轨迹训练 Actor-Critic。

```
感知编码器: o_t → z_t（潜状态）
世界模型（RSSM）: (z_t, a_t) → z_{t+1}（在潜空间演化）
奖励模型: z_t → r_t
Actor: z_t → a_t（在想象中训练）
Critic: z_t → V(z_t)（在想象中训练）
```

优点：
- 极高样本效率，几乎在所有任务上优于 Model-Free
- DreamerV3 实现了真正的通用性（Atari/DMControl/Minecraft 等）

局限：
- 机器人真实部署的精度要求难以保证（模型误差累积）
- 高频控制下潜空间动力学不稳定

### MBPO（Model-Based Policy Optimization, Janner et al. 2019）

**核心思想**：用神经网络集成模型（Ensemble of Neural Networks）生成短 rollout，与真实数据混合训练 SAC。

- 集成模型（5~7 个网络）检测不确定区域，避免过度利用错误模型
- 短 rollout（1~5 步）避免模型误差累积
- 真实数据 + 模型数据混合训练

在连续控制基准上，MBPO 用约 5% 的 SAC 样本量达到相同性能。

### PETS（Probabilistic Ensembles with Trajectory Sampling, Chua et al. 2018）

**核心思想**：不显式学习策略，直接用模型集成做 CEM（交叉熵方法）规划。

- 模型：概率神经网络集成（捕获认知不确定性 + 偶然不确定性）
- 规划：CEM 在模型中优化动作序列
- 无需策略训练，只需模型 + 规划器

特别适合数据稀少的真实机器人操作场景。

### TD-MPC / TD-MPC2

**核心思想**：结合 Temporal Difference（TD）价值学习和 Model Predictive Control。

- 学习隐空间动力学模型 + 价值函数
- 规划时用 MPPI 在潜空间搜索，用价值函数截断规划 horizon
- 在机器人操作和 locomotion 上都有强结果

---

## MBRL vs Model-Free RL 对比

| 维度 | MBRL | Model-Free RL |
|------|------|---------------|
| 样本效率 | ✅ 高（可用模型生成数据） | ❌ 低（需大量真实交互） |
| 渐近性能 | ⚠️ 受模型精度限制 | ✅ 理论上可达最优 |
| 实现复杂度 | ❌ 高（需学模型 + 策略） | ✅ 低 |
| 计算效率 | ❌ 推理时规划开销大 | ✅ 策略直接查询 |
| 在机器人上的应用 | 操作任务、真实机器人 | Locomotion（高频控制） |
| 代表算法 | Dreamer, MBPO, PETS | PPO, SAC, TD3 |

---

## 在机器人中的应用场景

### 适合 MBRL 的场景
- **操作任务**：接触动力学复杂，需要精细规划；数据稀少（真实机器人）
- **未知环境适应**：RMA 的 Adaptation Module 本质是隐式的模型识别
- **低频控制**：规划 horizon 不需要太长，模型误差不累积
- **样本稀缺的真实机器人学习**：PETS 类方法

### 不适合 MBRL 的场景
- **高频 locomotion 控制**（200~1000 Hz）：规划开销无法实时
- **高维视觉输入 + 接触丰富**：模型难以精确，误差在 rollout 中爆炸
- **需要极高渐近性能**：模型误差有上界，Model-Free 可以做到更好

---

## 参考来源

- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3, 2023) — 世界模型通用化
- Janner et al., *When to Trust Your Model: Model-Based Policy Optimization* (MBPO, 2019) — 短 rollout 混合训练
- Chua et al., *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models* (PETS, 2018) — 集成模型 + CEM 规划
- Hansen et al., *TD-MPC2: Scalable, Robust World Models for Continuous Control* (2023) — 潜空间规划 + TD 价值
- Sutton, *Integrated architectures for learning, planning, and reacting* (Dyna, 1990) — MBRL 经典框架
- **ingest 档案：** [sources/papers/model_based_rl.md](../../sources/papers/model_based_rl.md)

---

## 关联页面

- [Reinforcement Learning](./reinforcement-learning.md) — MBRL 是 RL 大类下的子方向，与 Model-Free 并列
- [Model Predictive Control (MPC)](./model-predictive-control.md) — 基于模型规划的经典控制方法，MBRL 的"控制论版"
- [Trajectory Optimization](./trajectory-optimization.md) — MBRL 规划阶段常用轨迹优化作为求解器
- [Optimal Control (OCP)](../concepts/optimal-control.md) — MBRL 的数学基础，动力学模型 + 代价函数
- [Sim2Real](../concepts/sim2real.md) — MBRL 的样本效率优势直接帮助真实机器人学习
- [Imitation Learning](./imitation-learning.md) — 可以和 IL 结合：用演示数据初始化模型

## 一句话记忆

> Model-Based RL 用"学习的世界模型"代替部分真实交互，通过在想象中规划和练习大幅提升样本效率——代价是模型精度的上限和更高的实现复杂度。
