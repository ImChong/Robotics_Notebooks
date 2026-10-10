# AlphaStar 官方研究资料（Google DeepMind）

> 一手项目来源归档（复核日期：2026-10-10）

- **Nature 论文：** [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://doi.org/10.1038/s41586-019-1724-z)
- **DeepMind 研究博客：** <https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/>
- **StarCraft II Learning Environment：** [google-deepmind/pysc2](https://github.com/google-deepmind/pysc2)
- **AlphaStar 源码包：** [google-deepmind/alphastar](https://github.com/google-deepmind/alphastar)
- **开源判定：** **部分开源。** PySC2 环境公开；AlphaStar 源码包公开通用 agent 架构及 offline RL/behavior-cloning 的数据读取、训练、评测脚本。仓库 README 明确写明**没有提供 online RL training code**；不可把它当作论文中完整在线 League 训练管线或原始 Grandmaster agent 的开源权重。
- **关联论文：** [StarCraft II Unplugged: Large Scale Offline Reinforcement Learning](https://openreview.net/pdf?id=Np8Pumfoty)（NeurIPS 2021）
- **实体页：** [AlphaStar](../../wiki/entities/paper-alphastar.md)

## 开源范围核验

官方 README 推荐 Linux + Python 3.9，提供 pip editable install 或 Bazel 构建；训练入口位于 `alphastar/unplugged/scripts/train.py`，数据集读取和评测由对应子目录提供。仓库声明其发布代码面向研究工具与离线 RL，不是完整线上 AlphaStar 产品。
