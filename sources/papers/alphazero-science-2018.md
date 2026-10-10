# A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play（AlphaZero，Science 2018）

> 一手论文归档（复核日期：2026-10-10）

- **论文：** Silver et al., *Science* 362(6419), 1140–1144 (2018)
- **Science DOI：** <https://doi.org/10.1126/science.aar6404>
- **arXiv 预印本：** <https://arxiv.org/abs/1712.01815>
- **DeepMind 官方解读与棋谱：** <https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/>
- **项目详情：** [AlphaZero 实体页](../../wiki/entities/paper-alphazero.md)
- **官方代码状态：** 截至复核日，未找到 DeepMind 官方训练/推理代码或权重；DeepMind 发布了论文开放版本和部分对局记录。

## 一手资料摘录

1. AlphaZero 将 AlphaGo Zero 的自我对弈算法推广到国际象棋、将棋和围棋，只提供游戏规则，不依赖人类棋谱或游戏专用启发式。
2. 系统使用同一类神经网络与 MCTS 搜索结构；每个游戏只需指定规则与动作空间，训练过程从随机策略起步。
3. 论文报告在其比赛预算中，AlphaZero 击败了 Stockfish 8（国际象棋）与 Elmo（将棋），并击败 AlphaGo Zero（围棋）；官方博客给出 1,000 盘国际象棋比赛中 155 胜、6 负、839 和棋等数字。

## 对 wiki 的映射

- [AlphaZero 实体页](../../wiki/entities/paper-alphazero.md)
- [强化学习历史](../../wiki/concepts/reinforcement-learning-history.md)
- [强化学习历史](../../wiki/concepts/reinforcement-learning-history.md)
