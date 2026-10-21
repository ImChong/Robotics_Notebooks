# Ropedia W0：跨本体运动-操作世界模型（arXiv:2610.26855）

> 来源归档（paper）

- **标题：** Ropedia W0: Scaling Loco-Manipulation World Models across Embodiments with Human Experience
- **类型：** paper / world model / loco-manipulation / cross-embodiment
- **arXiv：** <https://arxiv.org/abs/2610.26855>
- **项目页：** <https://ropedia.github.io/ropedia-w0/>
- **代码：** <https://github.com/Ropedia/ropedia-w0>
- **提交日期：** 2026-10-19
- **一句话说明：** Ropedia W0 将人类第一视角经验与多种机器人数据纳入共享运动学表示，训练跨本体的运动-操作世界基础模型。

## 摘要级要点

- 论文报告数据集 Xp-Data 包含 10,000 小时人类第一视角数据和 3,000 小时机器人数据。
- 共享表示用相机位姿编码移动、用投影骨架编码操作，支持人和不同机器人之间迁移。
- 作者报告在 DreamDojo、Ψ0、ω-HOME、Xp-Teleop、AgiBot 等基准上的评测，以及蒸馏后实时、分钟级 rollout。
- 仓库已公开推理代码；README 仍将更多 checkpoint、Xp-Teleop 数据、评估与后训练代码列为待发布。源码与模型权重、数据集许可不同，分别核对。

## 对 wiki 的映射

- [Ropedia W0 实体页](../../wiki/entities/paper-ropedia-w0.md)
- [世界动作模型](../../wiki/entities/unifolm-world-model-action.md)
- [生成式世界模型](../../wiki/methods/generative-world-models.md)
