# SAC 论文通读（最大熵 off-policy actor-critic 视角）

> 来源归档（原文通读 · 支撑 `wiki/comparisons/ppo-vs-sac.md`）

- **标题：** Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor（ICML 2018）；配套续作 *Soft Actor-Critic Algorithms and Applications*（arXiv:1812.05905）
- **作者：** Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine（UC Berkeley）
- **类型：** paper / rl / off-policy
- **原文：** <https://arxiv.org/abs/1801.01290>（v2，2018-08-08）；续作 <https://arxiv.org/abs/1812.05905>
- **入库日期：** 2026-11-07
- **一句话说明：** 在最大熵 RL 框架下给出 off-policy、随机策略的 actor-critic 算法 SAC：用回放池复用数据，用熵项换探索与稳定性；续作加入双 Q、自动温度调节并做了真机实验。
- **核对范围：** 通读 1801.01290 正文 §1–§6（含相关工作、软策略迭代推导与引理 1–2、定理 1、SAC 实用算法、对比实验与消融）及附录 D 超参表；附录 A–C 中的证明只对照结论，没有逐步重推。续作 1812.05905 通读 §5（自动熵调节）、§6 算法、§7 实验（含 Minitaur 真机四足、灵巧手 valve 旋转）、§8 结论与附录 D 超参表；其余与 v2 重叠的推导只对照。没有运行官方代码。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题与动机（1801 §1）：** model-free 深度 RL 的两个障碍是样本复杂度极高、超参数脆弱。TRPO/PPO/A3C 属 on-policy，每个梯度步都要新采样；DDPG 样本高效但对超参数极敏感、在 Humanoid 这类高维任务上常失败。
- **最大熵目标（式 1）：** J(π)=Σ_t E[r(s_t,a_t)+α·H(π(·|s_t))]，温度 α 控制随机性；α→0 恢复标准 RL 目标。论文声称带来三点好处：更广泛的探索、能保留多个近似最优模式、学习更快。
- **软策略迭代（§4.1）：** 软 Bellman 备份 T^π Q(s,a)=r+γ E[V(s')]，V(s)=E_a[Q(s,a)−log π(a|s)]；策略改进把 π 朝 exp(Q) 做 KL 信息投影。引理 1（软策略评估收敛）、引理 2（软策略改进单调）、定理 1（收敛到策略类内最优），均要求表格情形且 |A|<∞。
- **实用 SAC（§4.2，Algorithm 1）：** 策略 π_φ、软 Q 网络 Q_θ，以及（1801 版独有的）状态价值网络 V_ψ 及其指数滑动平均目标网络 V_ψ̄；策略用重参数化 a=f_φ(ε;s) 最小化 E[log π − Q]（式 12），比似然比估计方差更低；用两个 Q 取最小值缓解过估计（引自 TD3 思路）；每个环境步后做一个或多个梯度步，数据来自回放池。
- **对比实验（§5.1，图 1）：** Hopper/Walker2d/HalfCheetah/Ant/Humanoid-v1 与 rllab Humanoid（21 维动作），对 DDPG、PPO、SQL、TD3；SAC 在简单任务上持平、在难任务上大幅领先；DDPG 在 Ant、Humanoid 上不进步；SAC 比 PPO 学得快，原文归因于 PPO 为稳定需要大 batch。5 个种子，评估时用策略均值动作。
- **消融（§5.2）：** 同样的 SAC 去掉熵、改确定性策略后，各种子间方差明显变大（图 2），说明随机策略加熵对稳定性有帮助；SAC 对奖励尺度敏感（奖励尺度相当于温度，太小则策略近乎均匀、利用不了奖励，太大则早期快但后期探索不足）；目标网络平滑系数 τ 也做了敏感性实验。
- **超参（附录 D，表 1）：** Adam、学习率 3e-4、γ=0.99、回放池 10^6、2 层×256 隐单元、minibatch 256、τ=0.005（硬更新变体为每 1000 步更新）；奖励尺度按环境单独调。
- **续作：自动温度（1812.05905 §5）：** 把"期望熵≥H̄"作为约束，对偶变量即温度 α；J(α)=E[−α log π(a|s) − α H̄]，用梯度下降同步更新 α；附录 D 给出目标熵 H̄=−dim(A)（如 HalfCheetah 为 −6）。作者在脚注中说明 v1 里单独的价值网络后来发现不必要，续作只保留双 Q 与目标 Q。
- **续作：真机四足 Minitaur（§7.2）：** 8 个直驱电机的小型四足，仅在平地上从 160k 环境步（约 400 回合、约 2 小时）学会行走，无仿真、无预训练；训练与采样异步跑在工作站与机载 Jetson TX2 上；状态拼接当前和过去 5 步观测与动作以应对延迟引起的非马尔可夫性；未训练过的坡道、障碍与台阶上仍能通过。
- **续作：灵巧手（§7.3）：** 9 自由度三指手转动 valve，原始 RGB 图像输入需约 300k 步（约 20 小时）；输入 valve 位置则约 3 小时，对比同任务先前 PPO 报告的 7.4 小时。
- **结论与展望（1801 §6）：** 随机、熵最大化的算法有望改善鲁棒性与稳定性；未来方向包括信赖域等二阶信息和更具表达力的策略类。

## 对 wiki 的映射

- [`wiki/comparisons/ppo-vs-sac.md`](../../wiki/comparisons/ppo-vs-sac.md)
- [`wiki/methods/sac.md`](../../wiki/methods/sac.md)

## 参考来源

- 论文：<https://arxiv.org/abs/1801.01290>
- 续作：<https://arxiv.org/abs/1812.05905>
