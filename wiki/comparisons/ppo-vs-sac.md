---
type: comparison
tags: [ppo, sac, rl, policy-optimization, locomotion, manipulation, on-policy, off-policy]
status: complete
summary: "PPO 与 SAC 在机器人 RL 任务中的系统性对比：on-policy vs off-policy 权衡、样本效率、稳定性、超参数敏感度与适用场景。"
sources:
  - ../../sources/papers/policy_optimization.md
related:
  - ../methods/policy-optimization.md
  - ../methods/reinforcement-learning.md
  - ../tasks/locomotion.md
  - ../formalizations/gae.md
---

# PPO vs SAC：机器人 RL 算法选型

**背景**：PPO（Proximal Policy Optimization）和 SAC（Soft Actor-Critic）是机器人 RL 领域最主流的两种连续控制算法。两者都已在真实机器人上取得成功，但底层训练范式截然不同：PPO 是 on-policy 算法，依赖大批量并行采样；SAC 是 off-policy 最大熵算法，依赖经验回放提升样本效率。工程选型需根据任务类型、硬件资源和训练环境综合判断。

## 一句话定位

> PPO 是"稳定、易调、并行友好"的 on-policy 主力；SAC 是"样本高效、探索充分"的 off-policy 精兵——前者适合仿真大规模 locomotion，后者适合样本受限的精细操作。

---

## 核心算法思想

### PPO：近端策略优化

PPO 的核心是 **clip 机制**，防止每次梯度更新步子过大导致策略崩溃：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta)\, A_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\, A_t \right) \right]$$

其中 $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是重要性采样比，$\varepsilon$ 通常取 0.1–0.2。

关键思路：**每次 rollout 后做多个 mini-batch 更新，但通过 clip 约束更新幅度**——既充分利用采样数据，又不破坏策略稳定性。PPO 配合 **GAE（广义优势估计）** 做优势函数估计，进一步降低梯度方差。

### SAC：软演员-评论家

SAC 在标准 RL 目标上引入 **最大熵正则项**，鼓励策略在完成任务的同时保持尽量高的随机性：

$$J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha\, \mathcal{H}(\pi(\cdot|s_t)) \right]$$

其中 $\alpha$ 是温度参数（可自动调节），$\mathcal{H}$ 是策略熵。

关键思路：**用 Replay Buffer 存储历史转移数据，off-policy 反复利用每条经验**——大幅提升样本效率。双 Q 网络（Clipped Double-Q）抑制 Q 值高估，自动调温度让熵目标自适应。

---

## 核心维度对比

| 维度 | PPO | SAC | 备注 |
|------|-----|-----|------|
| **策略类型** | On-policy（需要新数据） | Off-policy（经验回放） | 决定训练架构的根本差异 |
| **样本效率** | 低（每条数据只用 K 次后丢弃） | 高（Replay Buffer 数据反复复用） | SAC 比 PPO 高 10–100x |
| **训练稳定性** | 高（clip 防止崩溃，曲线平滑） | 中（Q 过估计 / 温度调节不当时不稳） | PPO 在早期调试中更可靠 |
| **超参数敏感度** | 低（`clip_range`、`lr` 不敏感） | 中（`alpha`、网络结构、Buffer 大小均有影响） | PPO 更易上手 |
| **并行仿真适配** | 极好（on-policy 天然适合大批量并行） | 一般（Buffer 与并行采样结合工程复杂） | PPO 是 Isaac Lab / legged_gym 的默认选择 |
| **真机 fine-tune** | 差（需要大量新数据，样本消耗高） | 好（少量真实数据即可有效更新） | SAC 在真机 fine-tune 阶段有明显优势 |
| **连续动作空间** | 好（高斯策略输出，天然连续） | 极好（最大熵框架对连续控制探索更充分） | 两者均支持高维连续动作 |
| **离散动作空间** | 支持（Softmax 策略） | 不推荐（最大熵在离散空间优势弱） | 离散场景 PPO 更合适 |
| **典型应用场景** | Locomotion（仿真大规模训练） | Manipulation、精细操作、真机 RL | 见下方决策指南 |

---

## 训练流程对比

### PPO 典型流程

```
初始化策略 π_θ、价值网络 V_φ
重复：
  1. 用当前 π_θ 在 N 个并行环境中采集 T 步 rollout
  2. 用 GAE 计算每步优势估计 Â_t
  3. 对 rollout 数据做 K 个 epoch 的 mini-batch 梯度更新（clip 约束）
  4. 丢弃旧 rollout，用新策略重新采样
直到收敛
```

核心特征：**新鲜数据驱动**——每次采集完整 batch 再更新，更新后数据即废弃。

### SAC 典型流程

```
初始化策略 π_θ、Q 网络 Q_φ1, Q_φ2、Replay Buffer D
重复：
  1. 用当前 π_θ 执行动作，收集转移 (s, a, r, s') 存入 D
  2. 从 D 中随机采样 mini-batch
  3. 更新 Q 网络（最小化 Bellman 误差）
  4. 更新策略（最大化 Q + 熵）
  5. 自动调节温度 α（可选）
直到收敛
```

核心特征：**经验复用驱动**——每步收集一条数据即可触发更新，Buffer 中的旧数据反复利用。

---

## 超参数标准配置参考

### PPO 标准配置（locomotion，Isaac Lab）

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| `num_envs` | 4096（GPU 并行） | 越多越好，直到显存上限 |
| `n_steps` | 24–64 步 | 每个 env 每轮采集步数 |
| `clip_range` (ε) | 0.1–0.2 | 更大步子用 0.2，保守用 0.1 |
| `learning_rate` | 1e-4–3e-4 | 可线性衰减到 0 |
| `n_epochs` | 5–10 | 每轮 rollout 做几次 mini-batch 更新 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE λ，控制偏差-方差权衡 |
| `entropy_coef` | 0.001–0.01 | 防止策略过早收敛到确定性 |

### SAC 标准配置（manipulation，真机）

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| `buffer_size` | 1M–5M | 越大越稳，受内存限制 |
| `batch_size` | 256–1024 | 从 Buffer 中采样的 mini-batch 大小 |
| `learning_rate` | 3e-4 | 策略网络和 Q 网络通常同一学习率 |
| `tau` | 0.005 | 目标网络软更新系数 |
| `alpha` | 自动调节 | 设定目标熵为 -dim(action)，让 α 自适应 |
| `learning_starts` | 1000–10000 步 | Buffer 预热步数，先不更新 |
| `train_freq` | 1 步 / 1 次更新 | 实时更新；或 N 步后批量更新 |

---

## 何时选 PPO

**适合场景**：

1. **仿真大规模并行训练**：Isaac Lab / legged_gym 4096+ 环境并行，PPO on-policy 天然契合，吞吐量极高
2. **人形 / 足式 locomotion**：奖励设计清晰，训练步数充足（>500M），PPO 是社区默认选择
3. **首次实验 / 快速 reward 验证**：超参数少，收敛曲线平滑，容易定位问题
4. **需要可复现的 baseline**：PPO + legged_gym / Isaac Lab 是大量论文的标准 baseline
5. **运动风格学习（AMP）**：PPO + Adversarial Motion Prior 判别器是标准搭配

**代表工作**：
- Rudin et al., *Learning to Walk in Minutes* (2022) — legged_gym + PPO
- Kumar et al., *RMA* (2021) — Isaac Gym + PPO，真机适应
- Zhuang et al., *Robot Parkour Learning* (2023) — PPO 极限运动策略

---

## 何时选 SAC

**适合场景**：

1. **样本成本高（真机 RL / 低并行度）**：SAC 每条数据利用率远高于 PPO，适合真实机器人收集的宝贵数据
2. **精细操作 / 灵巧手任务**：最大熵框架探索更充分，适合高维精细控制
3. **仿真后真机 fine-tune**：先 PPO 仿真预训练，再 SAC 真机少量数据微调
4. **接触丰富的操作任务**：SAC 在 contact-rich manipulation 上通常优于 PPO
5. **需要多模态探索**：熵正则化防止策略过早坍缩到单一模式

**代表工作**：
- Haarnoja et al., *SAC: Off-Policy Maximum Entropy Deep RL* (2018) — SAC 原论文
- OpenAI *Learning Dexterous In-Hand Manipulation* (2019) — 灵巧手操作
- Smith et al., *Legged Robots that Keep on Learning* (2022) — 真机 SAC fine-tune

---

## 决策规则

```
你的训练环境是什么？
│
├── 仿真 + 大规模 GPU 并行（>1000 个环境）
│     └── → PPO（on-policy 并行采样效率无可比拟）
│
├── 真实机器人 / 样本成本高（<10K 次交互预算）
│     └── → SAC（off-policy 样本效率高 10–100x）
│
├── 任务类型是 locomotion（行走 / 奔跑 / 跳跃）
│     └── → PPO（社区标准，baseline 丰富，收敛稳定）
│
├── 任务类型是操作（抓取 / 装配 / 灵巧手）
│     └── → SAC（最大熵框架对精细操作探索更充分）
│
├── 首次实验 / 需要快速验证 reward 设计
│     └── → PPO（超参数少，曲线好读，出问题容易定位）
│
├── 需要运动参考 / 自然步态（AMP 框架）
│     └── → PPO + AMP（判别器 reward 驱动，底层优化器用 PPO）
│
└── 仿真预训练 → 真机迁移（两阶段）
      └── → 第一阶段 PPO（仿真大规模） + 第二阶段 SAC（真机 fine-tune）
```

---

## 混合策略：PPO 预训练 + SAC 微调

实践中越来越常见的两阶段方案：

```
阶段 1：仿真 PPO 训练（Isaac Lab，4096 envs，500M steps）
               ↓
        获得鲁棒初始策略（sim2real 可部署）
               ↓
阶段 2：真机 SAC fine-tune（真实机器人，<50K steps）
               ↓
        适配真实硬件动力学，消除残余 sim2real gap
```

优势：
- PPO 在仿真中用并行规模覆盖大量状态空间
- SAC 在真机上用样本效率弥补仿真 gap，避免大量真机交互

---

## 参考来源

- [sources/papers/policy_optimization.md](../../sources/papers/policy_optimization.md) — PPO / SAC 核心论文档案
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017) — PPO 原论文
- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor* (2018) — SAC 原论文
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning* (2022) — PPO locomotion 代表
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (2021) — PPO + 适应模块

---

## 关联页面

- [Policy Optimization](../methods/policy-optimization.md) — PPO / SAC 算法详细展开
- [Reinforcement Learning](../methods/reinforcement-learning.md) — RL 方法全局视角
- [Locomotion](../tasks/locomotion.md) — PPO 主要应用场景
- [GAE](../formalizations/gae.md) — PPO 的优势估计方法（广义优势估计）
- [PPO vs SAC for Robots（查询页）](../queries/ppo-vs-sac-for-robots.md) — 面向具体实践问题的快速决策指南

## 一句话记忆

> PPO 稳定、并行友好，是仿真 locomotion 的不二之选；SAC 样本高效、探索充分，是真机 fine-tune 和精细操作的首选——两者不是竞争关系，最优方案往往是先 PPO 再 SAC 的两阶段组合。
