# policy_optimization

> 来源归档（ingest）

- **标题：** Policy Optimization — PPO / SAC / TD3 及机器人应用
- **类型：** paper
- **来源：** arXiv / NeurIPS / ICLR / ICML
- **入库日期：** 2026-04-14
- **最后更新：** 2026-04-24
- **一句话说明：** 覆盖 model-free 策略优化核心算法，支撑 policy-optimization.md、reinforcement-learning.md 和 locomotion RL 应用页面。

## 核心论文摘录（MVP）

### 1) Proximal Policy Optimization Algorithms (Schulman et al., 2017)
- **链接：** <https://arxiv.org/abs/1707.06347>
- **核心贡献：** PPO：用 clip 约束代替 TRPO 的 KL 约束，在保证策略更新稳定性的同时大幅降低计算复杂度。在机器人 locomotion 中几乎是仿真训练的标准算法。
- **关键公式：** $L^{CLIP}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t,1-\varepsilon,1+\varepsilon)\hat{A}_t)]$
- **对 wiki 的映射：**
  - [Policy Optimization](../../wiki/methods/policy-optimization.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)
  - [RL Algorithm Selection](../../wiki/queries/rl-algorithm-selection.md)

### 2) Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning (Haarnoja et al., 2018)
- **链接：** <https://arxiv.org/abs/1801.01290>
- **核心贡献：** SAC：最大熵框架，自动在探索和利用间平衡，off-policy 高样本效率，适合真实机器人少样本学习。
- **关键公式：** $J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}[r(s_t,a_t) + \alpha\mathcal{H}(\pi(\cdot|s_t))]$
- **对 wiki 的映射：**
  - [Policy Optimization](../../wiki/methods/policy-optimization.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

### 3) Addressing Function Approximation Error in Actor-Critic Methods (Fujimoto et al., 2018)
- **链接：** <https://arxiv.org/abs/1802.09477>
- **核心贡献：** TD3：双 Critic 消除 Q 值过估计 + 延迟策略更新 + 目标策略平滑，在确定性策略下比 DDPG 稳定得多。
- **对 wiki 的映射：**
  - [Policy Optimization](../../wiki/methods/policy-optimization.md)

### 4) Trust Region Policy Optimization (Schulman et al., 2015)
- **链接：** <https://arxiv.org/abs/1502.05477>
- **核心贡献：** TRPO：通过 KL 散度约束限制策略更新幅度，是 PPO 的前身，理论更严格但计算更重。
- **对 wiki 的映射：**
  - [Policy Optimization](../../wiki/methods/policy-optimization.md)

### 5) Learning to Walk in Minutes Using Massively Parallel Deep RL (Rudin et al., 2022)
- **链接：** <https://arxiv.org/abs/2109.11978>
- **核心贡献：** Isaac Gym + PPO 的标志性工作，8192 并行环境 + 20 分钟训练出四足/双足步态，开启大规模并行 RL 训练范式。
- **对 wiki 的映射：**
  - [Locomotion](../../wiki/tasks/locomotion.md)
  - [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md)
  - [legged_gym](../../wiki/entities/legged-gym.md)
  - [curriculum-learning](../../wiki/concepts/curriculum-learning.md)
  - [reward-design](../../wiki/concepts/reward-design.md)

### 6) Bounded Ratio Reinforcement Learning (Ao et al., 2026)
- **链接：** <https://arxiv.org/abs/2604.18578>
- **项目页：** <https://bounded-ratio-rl.github.io/brrl/>
- **代码：** <https://github.com/bounded-ratio-rl/bounded_ratio_rl>
- **核心贡献：** 提出 BRRL（Bounded Ratio RL）框架，在有界 ratio 约束下给出解析最优策略，并据此构造 BPO（Bounded Policy Optimization）损失；论文给出单调改进理论保证，并在 MuJoCo / Atari / IsaacLab（含人形 locomotion）中报告了相对 PPO 的稳定性和性能优势。
- **关键机制：** 将 PPO 中启发式 clip 目标重新解释为“朝向有界 ratio 最优解”的近似优化，并建立 BRRL 与 trust region 方法、CEM 的联系。
- **对 wiki 的映射：**
  - [Policy Optimization](../../wiki/methods/policy-optimization.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [PPO vs SAC](../../wiki/comparisons/ppo-vs-sac.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)

## 当前提炼状态

- [x] PPO / SAC / TD3 / TRPO / Rudin et al. / BRRL 六条核心摘要
- [ ] 后续补：在不同机器人任务上三种算法的实测对比数据
