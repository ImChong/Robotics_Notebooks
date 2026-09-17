# WBC（全身控制）

**WBC** 负责把上层给出的多个任务目标（质心轨迹、末端位姿、姿态、接触力）与机器人的物理约束（动力学、关节限位、摩擦锥）一起写成一个优化问题，解出这一拍该发给每个关节的力矩。它是人形运控栈里承上启下的那一层：上面是 MPC / 规划 / 策略，下面是关节驱动器。

**站内入口：** [Whole-Body Control](../../../wiki/concepts/whole-body-control.md) · [WBC 实现指南](../../../wiki/queries/wbc-implementation-guide.md) · [WBC 调参](../../../wiki/queries/wbc-tuning-guide.md) · [什么时候用 WBC、什么时候用 RL](../../../wiki/queries/when-to-use-wbc-vs-rl.md)
