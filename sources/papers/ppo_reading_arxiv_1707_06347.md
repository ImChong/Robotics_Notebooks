# PPO 论文通读（RL 策略优化视角）

> 来源归档（原文通读 · 支撑 `wiki/methods/reinforcement-learning.md`）

- **标题：** Proximal Policy Optimization Algorithms
- **作者：** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov（OpenAI）
- **类型：** paper / rl / policy-gradient
- **原文：** <https://arxiv.org/abs/1707.06347>（2026-11 通过 arXiv PDF v2 获取全文）
- **一句话说明：** 提出用裁剪概率比的代理目标，在只用一阶优化的前提下获得接近 TRPO 的稳定性，并允许对同一批数据做多 epoch 小批量更新；是人形 / 足式 RL 最常用的 on-policy 算法出处。
- **核对范围：** 通读正文 §1–§7（含 §2 背景、§3 裁剪目标、§4 自适应 KL、§5 算法、§6.1–§6.4 实验）及附录 A 超参表；附录 B 的 49 款 Atari 学习曲线 / 分表仅浏览，未逐项核对。未阅读 Sutton & Barto 教材等其他综述，RL 通用概念仅依赖 wiki 页面既有内容。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **动机（§1）：** 当时主流方法的问题：DQN 在连续控制基准上未见良好表现（脚注 1）；vanilla policy gradient 数据效率与鲁棒性差；TRPO 较复杂，且与含噪声（dropout）或参数共享的结构不兼容。目标是同时做到可扩展、数据高效、鲁棒。
- **策略梯度（§2.1，式 1–2）：** 梯度估计 ĝ = Ê[∇θ log πθ(a|s) Â]；对同一轨迹做多步优化 L^PG 并无依据，经验上常导致破坏性的过大策略更新。
- **TRPO（§2.2）：** 在 KL 约束下最大化代理目标；理论上对应带系数 β 的 KL 惩罚，但单一 β 难以跨问题（甚至同一问题的不同阶段）通用，所以固定惩罚 + SGD 不够。
- **裁剪目标（§3，式 7）：** L^CLIP = Ê[min(r·Â, clip(r, 1−ε, 1+ε)·Â)]，r 为新旧策略概率比，ε 如 0.2；取 min 使其成为未裁剪目标的悲观下界；仅在会让目标变好时才忽略比率变化。
- **自适应 KL（§4）：** 目标 KL d_targ，d < d_targ/1.5 则 β←β/2，d > 1.5 d_targ 则 β←2β；实验中表现不如裁剪，作为基线保留。
- **完整目标与算法（§5，式 9、Algorithm 1）：** L = Ê[L^CLIP − c1·L^VF + c2·S[πθ]]（价值误差 + 熵奖励，用于策略 / 价值共享参数与探索）；优势用截断 GAE（式 11–12）；N 个并行 actor 各采 T 步，对 N·T 数据做 K epoch 小批量优化（常用 Adam）。
- **替代目标对比（§6.1，表 1）：** 7 个 MuJoCo 任务、各 100 万步、3 种子；无裁剪无惩罚 −0.39（HalfCheetah 上比随机策略更差），裁剪 ε=0.2 为 0.82（最佳），ε=0.1 / 0.3 为 0.76 / 0.70，自适应 KL 0.68–0.74，固定 KL 0.62–0.72。策略为 2×64 tanh MLP 输出高斯均值。
- **与其他算法对比（§6.2）：** 在多数连续控制环境上优于 TRPO、CEM、自适应步长 vanilla PG、A2C、A2C+trust region。
- **人形演示（§6.3）：** Roboschool 3D 人形的奔跑 / 转向（Flagrun）/ 被方块砸后起身（FlagrunHarder），训练 50M–100M 步；并提到同期 Heess 等用自适应 KL 版 PPO 学习 3D 机器人运动。
- **Atari（§6.4，表 2）：** 49 款游戏中，按"全程平均奖励"PPO 赢 30 款、ACER 18、A2C 1；按"末 100 回合"ACER 28、PPO 19、A2C 1（即 PPO 学得快，终局性能 ACER 略占优）。
- **超参（附录 A，表 3）：** MuJoCo：T=2048、Adam 3e-4、10 epoch、minibatch 64、γ=0.99、λ=0.95；Roboschool：T=512、15 epoch、minibatch 4096、32 / 128 actor。
- **结论（§7）：** 比 vanilla PG 只需少量代码改动，兼具信赖域方法的稳定性，适用于策略 / 价值联合架构。

## 对 wiki 的映射

- [`wiki/methods/reinforcement-learning.md`](../../wiki/methods/reinforcement-learning.md)
- [`wiki/methods/ppo.md`](../../wiki/methods/ppo.md)

## 参考来源

- 论文：<https://arxiv.org/abs/1707.06347>
