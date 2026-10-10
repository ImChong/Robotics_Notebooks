# Mastering the game of Go without human knowledge（AlphaGo Zero，Nature 2017）

> 一手论文归档（复核日期：2026-10-10）

- **论文：** Silver et al., *Nature* 550, 354–359 (2017)
- **DOI / 正文：** <https://doi.org/10.1038/nature24270>
- **DeepMind 官方解读：** <https://deepmind.google/blog/alphago-zero-starting-from-scratch/>
- **项目详情：** [AlphaGo Zero 实体页](../../wiki/entities/paper-alphago-zero.md)
- **公开补充材料：** Nature 页面提供补充信息与自我对弈/比赛棋谱。
- **官方代码状态：** 截至复核日，官方论文及项目页面未提供可运行训练代码或模型权重；社区实现不等于 DeepMind 官方实现。

## 一手资料摘录

1. AlphaGo Zero 仅从围棋规则开始，以强化学习自我对弈学习；训练不使用人类棋谱，也不使用传统围棋手工特征。
2. 一个残差网络同时输出着法策略和局面价值，MCTS 在策略先验与价值估计引导下搜索。
3. Nature 正文报告：训练三天后，AlphaGo Zero 以 100:0 击败先前发表的 AlphaGo 版本。对比应限定论文中的训练配置和计算预算。

## 对 wiki 的映射

- [AlphaGo Zero 实体页](../../wiki/entities/paper-alphago-zero.md)
- [强化学习 Runner 与自博弈](../../wiki/concepts/rl-runner.md)
- [深度强化学习游戏里程碑](../../wiki/concepts/deep-rl-game-milestones.md)
