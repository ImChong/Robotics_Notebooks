# Instant Episode Repetition（IER）（arXiv:2608.17347）

> 来源归档（ingest）

- **标题：** Repetition as Reinforcement: Enhancing Sample Efficiency via Instant Episode Repetition in Reinforcement Learning
- **缩写 / 框架：** **IER**（Instant Episode Repetition）
- **类型：** paper / reinforcement-learning / sample-efficiency / off-policy / continuous-control
- **arXiv：** <https://arxiv.org/abs/2608.17347>（PDF：<https://arxiv.org/pdf/2608.17347>）
- **会议：** Reinforcement Learning Conference（RLC）2026
- **代码：** <https://github.com/UoA-CARES/instant-episode-repetition>
- **作者：** Hoda Yamani、Yuning Xing、Koen van Rijnsoever、Bruce A. MacDonald、Henry Williams
- **机构：** 奥克兰大学（University of Auckland）/ CARES（Centre for Automation and Robotic Engineering Science）
- **入库日期：** 2026-08-20
- **一句话说明：** 在离策略 RL 交互环中，当 episode 刷新最高累积回报时，立即在环境中重放该动作序列 RN 次，而非仅在 replay buffer 被动复用。

## 开源状态（步骤 2.5）

- **论文：** 明确给出 GitHub URL。
- **仓库：** [UoA-CARES/instant-episode-repetition](https://github.com/UoA-CARES/instant-episode-repetition) 含 SAC/TD3 训练入口、`train_loops/ier/` 与 `configs/ier/` YAML。
- **结论：** **已开源**。

## 摘录 1：问题与动机（§1）

- **痛点：** 离策略 RL 样本效率低；Experience Replay / PER 只在 **策略更新** 时被动复用转移，不改变 **交互采样**；SIL 仍从 replay 采样高回报轨迹，agent 在环境中很少重复成功行为。
- **生物启发：** 奖励驱动重复巩固程序记忆；IER 在发现新高回报 episode 后 **立即** 重执行其动作序列。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-instant-episode-repetition.md`](../../wiki/entities/paper-instant-episode-repetition.md)；与 experience replay、SIL 对照。

## 摘录 2：方法（§3）

- **触发：** 若 \(R_{\mathrm{ep}}(\tau) > R_{\max}\)，存储动作序列 \(\mathbf{a}^*\)，进入 repetition mode。
- **交互：** 正常时 \(a_t \sim \pi_\theta(\cdot|s_t)\)；重复时 \(a_t = a_t^*\) 连续 **RN** 个 episode；之后回到策略采样。
- **学习：** 所有转移仍进 replay buffer，SAC/TD3 更新不变；**不改网络、损失或优化器**。
- **与 replay 区别：** 重放是在 **环境中再执行** 同一动作序列，初始状态/随机性不同 → 非严格轨迹复制。

**对 wiki 的映射：** 强调「交互层 plug-in」定位；RN 为关键超参。

## 摘录 3：实验（§4–5）

- **仿真：** MuJoCo（Ant/HalfCheetah/Humanoid/Hopper）+ DMC（Walker/Cheetah/Cartpole/Finger）；IER-SAC / IER-TD3 相对基线与 SIL 变体提升样本效率。
- **真机：** 双指操纵 **dynamic object translation**；IER 在真实接触/摩擦噪声下仍有效。
- **RN：** 论文扫 \(RN \in \{0,\ldots,7\}\)；中等 RN 通常最优，依任务与算法而异。

**对 wiki 的映射：** 写清适用 off-policy 连续控制 + 可定义 episode 级回报的场景。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-instant-episode-repetition.md`**；注册机构 **uoa**。
- 交叉更新 [强化学习](../../wiki/methods/reinforcement-learning.md)、[online vs offline RL](../../wiki/comparisons/online-vs-offline-rl.md) 选型。
