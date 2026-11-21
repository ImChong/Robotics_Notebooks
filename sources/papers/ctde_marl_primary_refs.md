# CTDE 多智能体强化学习（MAPPO / QMIX / MADDPG）核心奠基论文

> 来源归档（paper；2026-11-22 核对并通读）

- **合集定位：** 面向 CTDE vs 完全去中心化 MARL 对比的一手学术论文集，核对三篇最具代表性的 CTDE 原始论文。
- **论文列表：**
  1. Yu et al. (2022), *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (MAPPO) — <https://arxiv.org/abs/2103.01955>
  2. Rashid et al. (2018), *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning* — <https://arxiv.org/abs/1803.11485>
  3. Lowe et al. (2017/2020), *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (MADDPG) — <https://arxiv.org/abs/1706.02275>
- **入库日期：** 2026-11-22
- **通读：** 已完成
- **核对范围：** arXiv:2103.01955v4 (MAPPO)、arXiv:1803.11485v2 (QMIX)、arXiv:1706.02275v4 (MADDPG) 三篇 PDF 全文与结论。
- **一句话说明：** MAPPO、QMIX 与 MADDPG 从集中式值函数/critic 与单调价值分解出发，奠定了“集中式训练、分布式执行（CTDE）”在协作与混合博弈 MARL 中的主流基线。

## 核心摘录（面向 wiki 编译）

1. **MADDPG（Lowe et al., arXiv:1706.02275）：** 在 Actor-Critic 中为每个智能体引入可观测全局状态与所有智能体动作的集中式 critic，执行期 actor 仅凭局部观测输出动作，从而缓解多智能体策略同时演化导致的环境非平稳性。
2. **QMIX（Rashid et al., arXiv:1803.11485）：** 将联合动作价值 $Q_{\text{tot}}$ 通过混合超网络分解为个体 $Q_a$ 的单调函数（$\partial Q_{\text{tot}} / \partial Q_a \ge 0$），并利用全局状态生成权重以保持集中训练与去中心贪心执行（IGM）一致性。
3. **MAPPO（Yu et al., arXiv:2103.01955）：** 证明在合理配置值函数输入（全局状态与特征筛选）、裁剪比率、批大小与训练 epoch 的前提下，采用 CTDE 架构的同策略 PPO 在协作多智能体基准上即可达到与 off-policy QMIX 相当甚至更高的样本效率与收益。

## 对 wiki 的映射

- [`wiki/comparisons/ctde-vs-decentralized-marl.md`](../../wiki/comparisons/ctde-vs-decentralized-marl.md)
