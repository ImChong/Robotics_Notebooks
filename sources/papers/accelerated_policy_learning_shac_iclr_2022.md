# Accelerated Policy Learning with Parallel Differentiable Simulation（SHAC）

> 来源归档（ingest · paper）

- **作者：** Jie Xu, Viktor Makoviychuk, Yashraj Narang, Fabio Ramos, Wojciech Matusik, Animesh Garg, Miles Macklin
- **发表：** ICLR 2022
- **论文：** [OpenReview](https://openreview.net/forum?id=ZSKRQMvttc) · [PDF](https://openreview.net/pdf?id=ZSKRQMvttc) · [arXiv:2204.07137](https://arxiv.org/abs/2204.07137)
- **项目页：** <https://short-horizon-actor-critic.github.io/>
- **代码：** <https://github.com/NVlabs/DiffRL>
- **入库日期：** 2026-10-23
- **一句话说明：** SHAC 将并行可微仿真梯度接入短视界 actor-critic 策略学习，提升高维控制的样本效率。

## 核心摘录

1. **算法：** 可微仿真提供策略梯度；SHAC 用平滑 critic 近似长期回报，并截断反向传播窗口以缓解梯度消失或爆炸。
2. **并行：** 多个物理环境并行执行 rollout，提升采样与梯度计算吞吐。
3. **高维评测：** Humanoid MTU 实验报告高于 PPO 的样本与墙钟效率；具体数值依赖论文基准、硬件和实现。
4. **对 Wiki 的映射：** [SHAC 论文实体](../../wiki/entities/paper-shac.md)、[可微仿真](../../wiki/concepts/differentiable-simulation.md)

## 参考来源（原始）

- OpenReview：<https://openreview.net/forum?id=ZSKRQMvttc>
- 项目页：<https://short-horizon-actor-critic.github.io/>
- 官方代码：<https://github.com/NVlabs/DiffRL>
