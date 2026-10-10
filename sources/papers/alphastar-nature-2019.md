# Grandmaster level in StarCraft II using multi-agent reinforcement learning（AlphaStar，Nature 2019）

> 一手论文归档（复核日期：2026-10-10）

- **论文：** Vinyals et al., *Nature* 575, 350–354 (2019)
- **DOI / 正文：** <https://doi.org/10.1038/s41586-019-1724-z>
- **DeepMind 官方解读：** <https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/>
- **StarCraft II Learning Environment：** [PySC2 官方仓](https://github.com/google-deepmind/pysc2)
- **后续开源训练包：** [AlphaStar 官方仓](https://github.com/google-deepmind/alphastar) — 只覆盖通用架构和离线训练/评测，不含原始在线联赛训练实现。
- **实体页：** [AlphaStar](../../wiki/entities/paper-alphastar.md)

## 一手资料摘录

1. Nature 论文将 AlphaStar 描述为端到端神经网络智能体：通过人类对局回放的监督学习初始化，再通过多智能体强化学习与 League 训练提升对抗能力。
2. League 维护主策略与不同类型的 exploiter，使训练对手包含多样历史策略，减轻单一自博弈对手导致的策略脆弱和遗忘。
3. 论文报告 AlphaStar 在完整 StarCraft II 对战中达到三个种族的 Grandmaster 段位，并高于 99.8% 官方排名玩家。该结果来自论文设定的在线梯级评估，不是公开代码包的离线基准结果。

## 对 wiki 的映射

- [AlphaStar 实体页](../../wiki/entities/paper-alphastar.md)
- [PySC2 官方代码仓归档](../repos/alphastar-google-deepmind.md)
- [物理自博弈案例](../../wiki/entities/skild-physical-self-play.md)
