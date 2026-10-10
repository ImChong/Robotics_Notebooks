# Mastering the game of Go with deep neural networks and tree search（AlphaGo，Nature 2016）

> 一手论文来源归档（复核日期：2026-10-10）

- **论文：** Silver et al., *Nature* 529, 484–489 (2016)
- **DOI / 正文：** <https://doi.org/10.1038/nature16961>
- **Google Research Publications：** <https://research.google/pubs/mastering-the-game-of-go-with-deep-neural-networks-and-tree-search/>
- **DeepMind 官方回顾：** <https://deepmind.google/blog/innovations-of-alphago/>
- **项目详情：** [AlphaGo 实体页](../../wiki/entities/paper-alphago.md)
- **官方源码：** 截至复核日，论文与官方项目资料未提供 AlphaGo 训练/推理代码或权重；Nature 页面提供论文补充材料与正式比赛棋谱。第三方复现不等于官方实现。

## 一手资料摘录

1. AlphaGo 将策略网络、价值网络与蒙特卡洛树搜索结合：策略网络提出候选着法，价值网络评估局面，搜索在有限计算预算内挑选落子。
2. 训练由人类高手棋谱监督学习初始化，随后用自我对弈强化学习改进策略，并用自我对弈局面训练价值网络。
3. 论文报告：AlphaGo 对当时其他围棋程序胜率为 99.8%，并以 5:0 击败欧洲冠军 Fan Hui。这个结果对应论文规定的硬件和比赛协议。

## 对 wiki 的映射

- [AlphaGo：深度策略/价值网络与树搜索](../../wiki/entities/paper-alphago.md)
- [深度强化学习游戏里程碑](../../wiki/concepts/deep-rl-game-milestones.md)
- [强化学习](../../wiki/methods/reinforcement-learning.md)
