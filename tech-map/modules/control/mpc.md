# MPC（模型预测控制）

**MPC** 在每个控制周期里用模型向前滚动预测一段时间，在满足约束的前提下解一个优化问题，只执行第一步再重新求解。它是「有模型、要满足硬约束、还要实时」这类运控问题的默认答案——腿足机器人的落脚点规划、质心轨迹与接触力分配大多走这条路。

**站内入口：** [模型预测控制](../../../wiki/methods/model-predictive-control.md) · [非线性 MPC](../../../wiki/methods/nonlinear-model-predictive-control.md) · [MPC 调参指南](../../../wiki/queries/mpc-tuning-guide.md) · [求解器选型](../../../wiki/queries/mpc-solver-selection.md) · [MPC vs RL](../../../wiki/comparisons/mpc-vs-rl.md)
