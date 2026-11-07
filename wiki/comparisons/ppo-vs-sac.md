---
type: comparison
tags: [ppo, sac, rl, policy-optimization, locomotion, manipulation, on-policy, off-policy]
status: complete
updated: 2026-11-07
summary: "PPO 与 SAC 在机器人 RL 任务中的系统性对比：on-policy vs off-policy 权衡、样本效率、稳定性、超参数敏感度与适用场景；含 BRRL/BPO 与 FlashSAC 高维 scaling 选型；PPO/SAC 的超参数与结论已对照原论文核实。"
original:
  - ../../sources/papers/ppo_reading_arxiv_1707_06347.md
  - ../../sources/papers/sac_reading_arxiv_1801_01290.md
  - ../../sources/papers/learning_to_walk_in_minutes_reading_arxiv_2109_11978.md
sources:
  - ../../sources/papers/ppo_reading_arxiv_1707_06347.md
  - ../../sources/papers/sac_reading_arxiv_1801_01290.md
  - ../../sources/papers/learning_to_walk_in_minutes_reading_arxiv_2109_11978.md
  - ../../sources/personal/rl_runner_types.md
  - ../../sources/papers/policy_optimization.md
  - ../../sources/papers/intentional_streaming_rl.md
  - ../../sources/papers/flashsac_arxiv_2604_04539.md
  - ../../sources/blogs/wechat_robotshub_ppo_locomotion_fundamentals.md
  - ../../sources/blogs/wechat_pinkrobot_off_on_policy_rl_evolution_2026-09-15.md
related:
  - ../overview/deep-rl-off-on-policy-evolution.md
  - ../methods/flashsac.md
  - ../concepts/rl-runner.md
  - ../methods/ppo.md
  - ../methods/policy-optimization.md
  - ../methods/reinforcement-learning.md
  - ../methods/intentional-updates-streaming-rl.md
  - ../tasks/locomotion.md
  - ../formalizations/gae.md
  - ../methods/sac.md
  - ../methods/gae.md
---

# PPO vs SAC (vs BRRL/BPO)：机器人 RL 算法选型

**背景**：PPO（Proximal Policy Optimization）和 SAC（Soft Actor-Critic）是机器人 RL 领域最主流的两种连续控制算法。二者分属 On-policy 与 Off-policy 两条典型演进链的末端代表，完整历史坐标见 [深度 RL Off/On-policy 技术地图](../overview/deep-rl-off-on-policy-evolution.md)。两者都已在真实机器人上取得成功，但底层训练范式截然不同：PPO 是 on-policy 算法，依赖大批量并行采样；SAC 是 off-policy 最大熵算法，依赖经验回放提升样本效率。**BRRL / BPO (2026)** 是 PPO 的 on-policy 改写：它为 PPO 的启发式 clip 目标提供了一个解析最优解的解释，并在 MuJoCo / Atari / Isaac Lab 上报告了不低于 PPO 的稳定性与最终性能。

## 一句话定义

> PPO 是"稳定、易调"的 on-policy 主力；SAC 是"样本高效"的 off-policy 精兵；**BRRL/BPO** 是在 PPO 框架内换用有界比率解析解的 on-policy 变体，论文报告训练更平滑，并给出理论单调改进结论（针对其解析最优解）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PPO | Proximal Policy Optimization | 人形/足式 locomotion 中最常用的 on-policy 策略梯度算法 |
| SAC | Soft Actor-Critic | 连续控制常用的 off-policy 最大熵算法 |
| BRRL | Bounded Ratio Reinforcement Learning | 在有界重要性比率约束下给出解析最优策略的框架（2026） |
| BPO | Bounded Policy Optimization | BRRL 的实用损失：最小化与解析最优解的优势加权散度 |
| FlashSAC | Fast and Stable Soft Actor-Critic | 高维机器人 scaling 式 off-policy SAC（2026） |
| GAE | Generalized Advantage Estimation | PPO 常用的优势估计，λ 控制偏差—方差 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| TRPO | Trust Region Policy Optimization | 带信赖域约束的策略优化算法，PPO 的前身 |
| REDQ | Randomized Ensembled Double Q-learning | SAC 的扩展，支持更高的梯度步/环境步比（Smith 等真机微调所用） |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |
| Isaac Lab | NVIDIA Isaac Lab | 基于 Omniverse 的机器人学习训练框架 |
| legged_gym | Legged Gym | 足式机器人 RL 训练的常用开源框架 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |
| RMA | Rapid Motor Adaptation | 从历史轨迹隐式估计环境参数的快速运动自适应 |
| Isaac Gym | NVIDIA Isaac Gym | GPU 并行刚体仿真训练环境 |
| PG | Policy Gradient | 以回报梯度更新策略参数的基础方法族 |
| GPU | Graphics Processing Unit | 图形处理器，大规模并行仿真训练的算力基础 |

## 为什么重要

选错算法的代价不是"多调几天参"，而是整个数据管线的形态不同：

1. **PPO 把瓶颈放在仿真吞吐。** Rudin 等人（2022）用 4096 个并行机器人、批大小 98304，在 20 分钟内训出可部署的 ANYmal 行走策略；on-policy 方法正好吃到 GPU 并行的红利。
2. **SAC 把瓶颈放在每条样本的价值。** Haarnoja 等人在真机 Minitaur 四足上只用约 16 万环境步（约 2 小时）就从零学会行走，在灵巧手 valve 任务上输入位置时约 3 小时学会（对比同任务先前 PPO 报告的 7.4 小时）。样本昂贵时这一差距决定可行性。
3. **选型会连带决定其他设计：** 是否需要回放池、是否需要大并行、奖励能否尺度归一、真机能否在线微调。

## 核心原理

### PPO：近端策略优化

PPO（Schulman 等，2017）的核心是 **clip 机制**，防止每次梯度更新步子过大导致策略崩溃：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta)\, A_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\, A_t \right) \right]$$

其中 $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是重要性采样比。论文在 7 个 MuJoCo 任务上比较各种替代目标：无裁剪无惩罚平均归一化得分 $-0.39$，$\varepsilon=0.2$ 的裁剪最好（0.82），$\varepsilon=0.1/0.3$ 分别为 0.76/0.70，自适应 KL 为 0.68–0.74。完整目标还含价值误差与熵奖励：$L=\hat{\mathbb{E}}[L^{\text{CLIP}}-c_1L^{VF}+c_2S[\pi_\theta]]$。

关键思路：**每次 rollout 后做多个 epoch 的 mini-batch 更新，但通过 clip 约束更新幅度**——既充分利用采样数据，又不破坏策略稳定性。优势用截断的 [GAE](../methods/gae.md) 估计，进一步降低梯度方差。

### BRRL / BPO：有界策略优化 (2026)

Ao 等人（arXiv:2604.18578）指出 PPO 的 clip 目标并非直接由它要近似的信赖域公式推出。BRRL 把"新旧策略比率落在 $(1-\epsilon,1+\epsilon)$"作为约束，加入对数势垒式正则（权重 $\lambda$），得到解析最优解

$$\pi^*(a|s)=\Big(1+\epsilon\tanh\frac{\tilde A^{\pi_0}(s,a)}{2\lambda}\Big)\pi_0(a|s),$$

其中 $\tilde A$ 以"软中位数"为基线；论文证明该解保证单调性能提升。BPO 是面向参数化策略的实用损失：最小化与该解析解的优势加权散度，并给出所得策略期望性能的下界；论文称它给 PPO 损失提供了新解释，并联系到 CEM。**BPO 仍是 on-policy 算法**（代码在 Stable Baselines3 与 RSL-RL 的 PPO 框架里改损失，沿用 GAE），"与 off-policy 兼容"在论文里没有出现；论文中的 off-policy 基线（SAC、DQN）只是对照。

### SAC：软演员-评论家

SAC（Haarnoja 等，2018）在标准 RL 目标上引入 **最大熵正则项**，鼓励策略在完成任务的同时保持尽量高的随机性：

$$J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha\, \mathcal{H}(\pi(\cdot|s_t)) \right]$$

其中 $\alpha$ 是温度参数，$\mathcal{H}$ 是策略熵。$\alpha\to 0$ 时恢复标准目标。续作（arXiv:1812.05905）把"期望熵不低于目标 $\bar{\mathcal H}$"作为约束，对偶变量即温度，用 $J(\alpha)=\mathbb{E}[-\alpha\log\pi(a|s)-\alpha\bar{\mathcal H}]$ 自动调节，目标熵取 $-\dim(\mathcal A)$；原 ICML 版本里温度是手工固定的，对奖励尺度很敏感（奖励尺度等价于温度）。

关键思路：**用 Replay Buffer 存储历史转移数据，off-policy 反复利用每条经验**——大幅提升样本效率。两个 Q 网络取最小值抑制过估计（沿用 TD3 的做法），策略用重参数化技巧最小化 $\mathbb{E}[\log\pi - Q]$，评估时取策略均值动作。论文的消融显示：同样的 SAC 去掉熵、改确定性策略后，不同随机种子之间的方差明显变大，说明随机策略加熵有助于稳定。

### 核心维度对比

| 维度 | PPO | SAC | BRRL / BPO |
|------|-----|-----|------------|
| **策略类型** | On-policy | Off-policy | On-policy（PPO 框架内换损失） |
| **样本效率（每环境步）** | 低：论文称 SAC 学得更快，因 PPO 需大 batch 才稳 | 高 | 论文未宣称优于 PPO；样本效率提升被列为未来工作 |
| **挂钟效率（大规模并行）** | 极好：Rudin 20 分钟 | 取决于实现；FlashSAC 在高维任务上追平甚至反超 | 同 PPO |
| **训练稳定性** | 高（clip 约束步长） | 论文报告跨种子一致性好（对比 DDPG） | 论文报告 Isaac Lab 四项任务训练更平滑 |
| **超参数敏感度** | 低（社区默认值通用） | 中：奖励尺度/温度敏感；v2 自动温度缓解 | 中：新增 $\lambda$ 与损失权重；论文称对熵系数更敏感 |
| **并行仿真适配** | 极好 | 一般 | 极好 |
| **真机 fine-tune** | 差 | 好（Minitaur 真机从零学会行走） | 未见真机验证 |
| **典型应用场景** | Locomotion（仿真大规模） | Manipulation、真机 RL | Locomotion、追求平滑收敛 |

> 上表"每环境步"与"挂钟"必须分开看：样本效率是 SAC 的传统优势，但在 GPU 万级并行下，吞吐更大的 PPO 往往挂钟更快，直到 FlashSAC 这类 scaling 配方出现。

### 训练流程对比

**PPO 典型流程**

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

**SAC 典型流程**

```
初始化策略 π_θ、Q 网络 Q_φ1, Q_φ2（及目标网络）、Replay Buffer D
重复：
  1. 用当前 π_θ 执行动作，收集转移 (s, a, r, s') 存入 D
  2. 从 D 中随机采样 mini-batch
  3. 更新两个 Q 网络（最小化软 Bellman 误差，目标用两个目标 Q 的最小值）
  4. 更新策略（最小化 log π − Q，重参数化）
  5. 更新温度 α（v2 自动调节）并软更新目标网络（τ=0.005）
直到收敛
```

核心特征：**经验复用驱动**——每个环境步后做一个或多个梯度步，Buffer 中的旧数据反复利用。

**流式 RL 中的 Intentional Policy Gradient**：[Intentional Updates（流式 RL）](../methods/intentional-updates-streaming-rl.md)（arXiv:2604.19033）在 **batch=1、无 replay** 设定下，用 intentional 步长作为 **PPO/TRPO 式信任域** 的廉价流式代理——与 PPO clip 解决同一「参数步长 ≠ 函数输出变化」痛点。

## 工程实践

### 超参数参考

**PPO（locomotion，Rudin 等 2022 的实际取值）**

| 超参数 | Rudin 论文值 | 说明 |
|--------|--------------|------|
| 并行机器人数 | 4096 | 并非越多越好：机器人太多时每台的时间视野过短，性能骤降（论文 §4.1），4096 台、batch 约 98304 为文中选用点 |
| 每环境步数 | 24 | 批大小 = 4096×24；论文指出少于约 25 步（约 0.5 s 仿真时间）会让依赖时间连贯轨迹的 GAE 难以收敛 |
| mini-batch | 24576（4096×6） | 论文发现并行场景下 mini-batch 取数万样本反而更稳 |
| epoch | 5 | 每轮 rollout 的 mini-batch 更新轮数（PPO 原论文 MuJoCo 为 10） |
| `clip_range` (ε) | 0.2 | 原论文中 0.2 最优 |
| 学习率 | 自适应（目标 KL 0.01） | 原 PPO 论文固定 3e-4 |
| 熵系数 | 0.01 | |
| $\gamma$ / $\lambda$ | 0.99 / 0.95 | GAE，和 PPO 原论文一致 |

另可参考 [GAE](../methods/gae.md) 页的 $\gamma,\lambda$ 取值讨论。社区常把 `n_steps` 设到 24–64、学习率 1e-4–3e-4 并线性衰减，这类经验区间未在本次入库中逐项核对，以论文值为准。

**SAC（Haarnoja 等 2018/2019 论文值）**

| 超参数 | 论文值 | 说明 |
|--------|--------|------|
| 优化器 / 学习率 | Adam，3e-4 | 策略、Q 网络、温度共用 |
| `buffer_size` | $10^6$ | |
| `batch_size` | 256 | |
| 网络 | 2 层 × 256 ReLU（所有网络） | |
| $\gamma$ | 0.99 | |
| `tau` | 0.005 | 目标网络软更新系数 |
| 目标熵 | $-\dim(\mathcal A)$（如 HalfCheetah −6） | 自动温度；原版需按环境手调奖励尺度 |
| 更新频率 | 每环境步 1 次梯度步 | 论文另给硬目标更新变体（每 1000 步覆盖目标网络）：除人形任务外每环境步做 4 次梯度步，人形任务做 1 次 |

`learning_starts`、`train_freq` 等参数来自常见实现库（如 Stable-Baselines3）的惯例，论文未规定，视为工程经验。

### 何时选 PPO

**适合场景**：

1. **仿真大规模并行训练**：Isaac Lab / legged_gym 4096+ 环境并行，PPO on-policy 天然契合，吞吐量极高
2. **人形 / 足式 locomotion**：奖励设计清晰，PPO 是社区默认选择（Rudin 等的行走策略训练用到约 1500 次策略更新、不到 20 分钟）
3. **首次实验 / 快速 reward 验证**：超参数少，收敛曲线平滑，容易定位问题
4. **需要可复现的 baseline**：PPO + legged_gym / Isaac Lab 是大量论文的标准 baseline
5. **运动风格学习（AMP）**：PPO + Adversarial Motion Prior 判别器是标准搭配

**代表工作**：
- Rudin et al., *Learning to Walk in Minutes* (2022) — legged_gym + PPO
- Kumar et al., *RMA* (2021) — PPO 训练特权基策略，真机适应
- Zhuang et al., *Robot Parkour Learning* (2023) — PPO 训练专项技能并微调
- OpenAI, *Learning Dexterous In-Hand Manipulation* (2019/2020) — **PPO**（论文明确使用 PPO，沿用 OpenAI Five 的分布式实现）；旧版本此处误列在 SAC 下

### 何时选 SAC

**适合场景**：

1. **样本成本高（真机 RL / 低并行度）**：SAC 每条数据利用率远高于 PPO，适合真实机器人收集的宝贵数据
2. **灵巧手 / 精细操作**：论文在 9 自由度三指手 valve 旋转上，输入位置约 3 小时、原始 RGB 约 20 小时学会
3. **仿真后真机 fine-tune**：先仿真预训练，再用 SAC 家族（如 REDQ）真机少量数据微调——Smith 等人的 A1 四足系统即是这一思路
4. **接触丰富的操作任务**：通常认为 SAC 优于 PPO，这属于社区经验，论文未对该论断做专门实验
5. **需要多模态探索**：熵正则化防止策略过早坍缩到单一模式（论文称最大熵目标可保留多个近最优模式）

**代表工作**：
- Haarnoja et al., *SAC* (2018) 及续作 *SAC Algorithms and Applications* (2018/2019) — Minitaur 真机 + 灵巧手
- Smith et al., *Legged Robots that Keep on Learning* (2021/2022) — 仿真预训练后真机用 REDQ（SAC 的扩展）持续微调 A1 四足

### 何时选 BRRL / BPO

**论文支持的场景**：

1. **Isaac Lab 复杂 locomotion（4096 环境）**：在 Go1/Anymal-C/G1/H1 粗糙地形四项任务上，G1-rough 明显高于 PPO 的性能上限，Go1 与 H1 略高，四项任务训练都更稳定、更平滑（5 个种子）。
2. **需要可解释性与理论保证**：理论对象是解析最优解的单调改进与 BPO 的性能下界；BPO 实际算法与理论解之间仍有近似误差，不应读成"BPO 每步单调提升"。
3. **PPO 末期振荡时可以一试**：这是对论文稳定性结果的推断（推测），非论文直接结论。

**需要警惕**：MuJoCo Humanoid-v4 上 BPO 跨种子方差大，且表 1 中 SAC 在 Ant-v4 与 Humanoid-v4 上都高于 BPO 和 PPO；BPO 对熵系数更敏感，论文建议在很多环境里把熵权重降到约 PPO 最优值的十分之一；论文尚无真机验证。

### 何时选 FlashSAC

**适合场景**（Kim et al., 2026，arXiv:2604.04539；详见 [FlashSAC 方法页](../methods/flashsac.md)）：

1. **高 DoF 状态控制**：人形（G1/H1）、灵巧手等 PPO 渐近与墙钟均落后的任务（论文图 1b：高维任务上渐近回报与挂钟都明显优于 PPO）
2. **高并行仿真 + 追墙钟 sim-to-real**：G1 盲行走平地约 **20 min** vs PPO **~3 h**；粗糙地形台阶约 **4 h** vs PPO **~20 h**（同一 sim-to-real 管线）
3. **希望 off-policy 复用 replay 且要渐近性能**：相对 FastSAC 约 0.2M 参数的小网络配方，FlashSAC 用 2.5M 参数、6 层网络加范数约束来兼顾上限
4. **视觉 DMControl 等低吞吐环境**：在 8 个任务、1M 步设定下渐近性能持平或高于所有基线，且挂钟收敛更快；DrQ-v2 样本高效但不稳定

**不太适合**：

- 低 DoF 四足/夹爪且仅需成熟 PPO baseline、对墙钟不敏感（论文图 1a：低维任务上与 PPO 相当）
- 极低并行、极小 buffer 且不愿调低 batch/UTD 的设定

**资料**：
- 论文：<https://arxiv.org/abs/2604.04539>
- 项目页：<https://holiday-robot.github.io/FlashSAC/>
- 代码：<https://github.com/Holiday-Robot/FlashSAC>
- FastSAC：[Learning Sim-to-Real Humanoid Locomotion in 15 Minutes](../entities/paper-notebook-learning-sim-to-real-humanoid-locomotion-in-15-m.md)

### 决策规则

```
你的训练环境是什么？
│
├── 仿真 + 大规模 GPU 并行（>1000 个环境）
│     ├── 追求成熟生态与快速 Baseline → PPO
│     ├── 高 DoF 人形/灵巧手且追墙钟与渐近 → FlashSAC
│     └── 想试更平滑的 PPO 式更新 → BRRL / BPO
│
├── 真实机器人 / 样本成本高（<10K 次交互预算）
│     ├── 经典 off-policy 微调 → SAC（或 REDQ 类高 UTD 变体）
│     └── 高维 sim-to-real 端到端训练 → FlashSAC（G1 行走等已验证）
│
├── 任务类型是 locomotion（行走 / 奔跑 / 跳跃）
│     ├── 社区默认 / 成熟生态 → PPO 或 BPO
│     └── 高 DoF 人形盲行走、楼梯等 → FlashSAC（论文约 9× 平地 / 5× 台阶墙钟 vs PPO）
│
├── 任务类型是操作（抓取 / 装配 / 灵巧手）
│     └── → SAC 或 FlashSAC（样本成本高 / 高维）；稀疏奖励可叠 HER
│
├── 首次实验 / 需要快速验证 reward 设计
│     └── → PPO（超参数少，曲线好读，出问题容易定位）
│
└── 仿真预训练 → 真机迁移（两阶段）
      └── → 第一阶段仿真 PPO/BPO + 第二阶段 SAC 家族真机 fine-tune
```

### 混合策略：仿真预训练 + SAC 家族微调

实践中越来越常见的两阶段方案：

```
阶段 1：仿真大规模并行训练（如 PPO，Isaac Lab，4096 envs）
               ↓
        获得鲁棒初始策略（sim2real 可部署）
               ↓
阶段 2：真机 off-policy fine-tune（SAC / REDQ 等，少量真机数据）
               ↓
        适配真实硬件动力学，消除残余 sim2real gap
```

优势：
- 仿真并行规模覆盖大量状态空间
- off-policy 在真机上用样本效率弥补仿真 gap，避免大量真机交互

两点说明：Smith 等人的系统确实是"仿真预训练 + 真机 REDQ 微调"，但本次核对未确认其仿真预训练使用的是 PPO；页面此前给出的"500M 步 + 50K 真机步"数字没有原文依据，已删除（推测：具体步数需按任务试验）。

## 局限与风险

- **PPO 的局限：** 数据利用率低（更新后即丢弃），难以用于样本昂贵的真机在线学习；对 GAE 的时间连贯性有要求（每环境步数太少会难收敛）。
- **SAC 的局限：** 在大规模并行仿真中不容易吃满吞吐；原版 SAC 对奖励尺度敏感；Q 函数过估计需要靠双 Q 缓解；论文实验以 MuJoCo 等低维到中维任务为主，高维 sim-to-real 的结论要看 FlashSAC 类后续工作。
- **BRRL/BPO 的局限：** 论文自述未来方向包括样本效率、约束 MDP 与生成式策略微调；目前没有真机实验，且在 MuJoCo Humanoid-v4 上存在跨种子高方差。
- **算法之外的混杂因素：** 本页比较多来自不同论文、不同仿真器与不同实现。PPO 的稳定性、SAC 的样本效率结论都依赖于奖励设计与网络规模，直接横向比大小要谨慎。
- **FlashSAC 数字的条件：** "20 分钟 vs 3 小时"是同一 sim-to-real 管线下论文报告的结果，对应 G1 盲行走平地任务，并非任何机器人上的通用加速比。

## 关联页面

- [Policy Optimization](../methods/policy-optimization.md) — PPO / SAC 算法详细展开
- [PPO（近端策略优化）](../methods/ppo.md) — PPO 方法专页（clip 代理目标 / on-policy）
- [FlashSAC（快速稳定 SAC）](../methods/flashsac.md) — 高维 off-policy scaling（2026）
- [SAC（软演员-评论家）](../methods/sac.md) — SAC 方法专页（最大熵 / off-policy / 自动调温度）
- [Reinforcement Learning](../methods/reinforcement-learning.md) — RL 方法全局视角
- [RL Runner（训练循环编排）](../concepts/rl-runner.md) — on-policy「采完即丢」vs off-policy Replay 的循环差，先于 clip/熵
- [Locomotion](../tasks/locomotion.md) — PPO 主要应用场景
- [GAE](../methods/gae.md) — PPO 的优势估计方法（广义优势估计）；形式化推导见 [GAE 形式化](../formalizations/gae.md)
- [PPO vs SAC for Robots（查询页）](../queries/ppo-vs-sac-for-robots.md) — 面向具体实践问题的快速决策指南

## 参考来源

- [PPO 论文归档（原文通读）](../../sources/papers/ppo_reading_arxiv_1707_06347.md) — 裁剪目标、替代目标对比、超参
- [SAC 论文归档（原文通读）](../../sources/papers/sac_reading_arxiv_1801_01290.md) — 最大熵目标、软策略迭代、自动温度、Minitaur / 灵巧手真机
- [Learning to Walk in Minutes 论文归档（原文通读）](../../sources/papers/learning_to_walk_in_minutes_reading_arxiv_2109_11978.md) — 4096 并行、PPO 超参、批大小权衡
- [RobotsHub：万字解析运控 PPO](../../sources/blogs/wechat_robotshub_ppo_locomotion_fundamentals.md) — 并行仿真下为何 PPO 压过 SAC 的样本效率
- [sources/papers/policy_optimization.md](../../sources/papers/policy_optimization.md) — PPO / SAC / BRRL 等核心论文档案
- [sources/papers/intentional_streaming_rl.md](../../sources/papers/intentional_streaming_rl.md) — 流式 intentional PG
- [sources/papers/flashsac_arxiv_2604_04539.md](../../sources/papers/flashsac_arxiv_2604_04539.md) — FlashSAC 论文档案
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017) — <https://arxiv.org/abs/1707.06347>
- Haarnoja et al., *Soft Actor-Critic* (2018) — <https://arxiv.org/abs/1801.01290>；续作 <https://arxiv.org/abs/1812.05905>
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning* (2022) — <https://arxiv.org/abs/2109.11978>
- Ao et al., *Bounded Ratio Reinforcement Learning* (2026) — <https://arxiv.org/abs/2604.18578>（本次对照了摘要、引言、定理 4.1、实验 §5.1–§5.2、消融与结论，非逐页通读）
- Kim et al., *FlashSAC* (2026) — <https://arxiv.org/abs/2604.04539>（本次对照了摘要、引言、§5.3–§5.4 与关键数字，非逐页通读）
- Smith et al., *Legged Robots that Keep on Learning* (2021) — <https://arxiv.org/abs/2110.05457>（核对了摘要、系统概述与 REDQ 部分）
- OpenAI, *Learning Dexterous In-Hand Manipulation* — <https://arxiv.org/abs/1808.00177>（核对了所用算法为 PPO）
- Kumar et al., *RMA* (2021) — 以 [本库 RMA 论文归档](../../sources/papers/rma_arxiv_2107_04034.md) 为准

## 推荐继续阅读

- [OpenAI Spinning Up：PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) 与 [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html) 的官方讲解与伪代码
- [Learning to Walk in Minutes（arXiv:2109.11978）](https://arxiv.org/abs/2109.11978) — 大规模并行 PPO 的批大小与时间视野权衡
- [Soft Actor-Critic Algorithms and Applications（arXiv:1812.05905）](https://arxiv.org/abs/1812.05905) — 自动温度与真机案例

## 一句话记忆

> PPO 稳定、并行友好，是仿真 locomotion 的默认选择；SAC 样本高效、探索充分，是真机 fine-tune 和精细操作的首选；高维任务上要看 FlashSAC。两者不是竞争关系，常见做法是先仿真大规模训练再在真机上用 off-policy 微调。
