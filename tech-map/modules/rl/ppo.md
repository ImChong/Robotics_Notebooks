# PPO

**PPO** 用一个裁剪过的目标函数限制每次更新的策略变化幅度，换来「调参不那么容易崩」的稳定性。它算法上不是最先进的，却是机器人 RL 里事实上的默认起点：并行仿真友好、超参相对鲁棒、几乎所有腿足运控的开源训练栈都以它为基线。

**站内入口：** [PPO](../../../wiki/methods/ppo.md) · [PPO vs SAC](../../../wiki/comparisons/ppo-vs-sac.md) · [RL 算法选型](../../../wiki/queries/rl-algorithm-selection.md) · [超参数指南](../../../wiki/queries/rl-hyperparameter-guide.md)
