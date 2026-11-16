# Actuation and Control of Passive Tendon-Driven Soft Extensions Through Rigid Manipulator Motion：作者代码核对

- **类型：** 官方代码仓
- **代码：** https://github.com/nachocz/passive_soft_extension_control
- **论文：** https://arxiv.org/abs/2611.15386
- **公开入口：** `python examples/run_scenario.py S2`
- **相关实现：** `softext 与约束控制器`
- **产物：** 仿真轨迹与图表
- **核对范围：** 官方 README 及对应启动/评测/模块目录的静态核查，未运行实际训练或硬件。
- **通读：** 已完成
- **wiki：** [论文详情](../../wiki/entities/paper-passive-soft-extensions-2611-15386.md)

**范围限制：** 辨识映射只在探测过的腱索跨度范围有效，冗余度与支点约束会限制可驱动的形变方向。
