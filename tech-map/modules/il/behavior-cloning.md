# Behavior Cloning（行为克隆）

**BC** 把「学策略」直接当成监督学习：收集专家的「状态 → 动作」数据对，训一个网络去拟合。它是模仿学习里门槛最低的一条路，也是 Diffusion Policy、VLA 等后续方法的共同底座；主要代价是分布漂移——一旦机器人走到演示没覆盖过的状态，误差会累积。

**站内入口：** [Behavior Cloning](../../../wiki/methods/behavior-cloning.md) · [BC 损失函数](../../../wiki/formalizations/behavior-cloning-loss.md) · [模仿学习](../../../wiki/methods/imitation-learning.md) · [操作任务怎么用 IL](../../../wiki/queries/il-for-manipulation.md)
