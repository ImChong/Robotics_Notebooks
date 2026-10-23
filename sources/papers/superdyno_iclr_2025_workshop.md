# Scalable Humanoid Whole-Body Control via Differentiable Neural Network Dynamics（SuperDyno）

> 来源归档（ingest · paper）

- **作者：** Yu Lei, Zhengyi Luo, Tairan He, Jinkun Cao, Guanya Shi, Kris Kitani
- **机构：** Carnegie Mellon University Robotics Institute
- **发表：** ICLR 2025 Workshop on World Models
- **论文：** [OpenReview](https://openreview.net/forum?id=FPgKt7bA8w) · [PDF](https://openreview.net/pdf?id=FPgKt7bA8w)
- **论文所列项目页：** <https://dynamics-humanoid.github.io/dynamics-humanoid/>
- **代码：** 截至 2026-10-23 未找到官方源码仓库；所列项目页当前返回 404。
- **入库日期：** 2026-10-23
- **一句话说明：** 学习可微神经动力学，并结合监督学习与强化学习，扩展人形全身动作跟踪。

## 核心摘录

1. **模型：** 从仿真轨迹学习神经网络动力学，并用可微模型预测状态、将目标梯度传回策略。
2. **训练：** 将监督学习与强化学习结合；论文报告相较先前方法有更高样本效率和更低跟踪误差。
3. **规模：** 作者报告方法可扩展到数万段动作片段，并评测稀疏奖励任务与模型跨任务迁移。
4. **对 Wiki 的映射：** [SuperDyno 论文实体](../../wiki/entities/paper-superdyno.md)、[可微仿真](../../wiki/concepts/differentiable-simulation.md)

## 参考来源

- OpenReview：<https://openreview.net/forum?id=FPgKt7bA8w>
- PDF：<https://openreview.net/pdf?id=FPgKt7bA8w>
- 论文所列项目页：<https://dynamics-humanoid.github.io/dynamics-humanoid/>
