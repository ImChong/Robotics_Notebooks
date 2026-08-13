# FloBaRoID（kjyv/FloBaRoID）

> 来源归档

- **标题：** FloBaRoID — FLOating BAse RObot dynamical IDentification
- **类型：** repo
- **来源：** 原 IIT Advanced Robotics（WALK-MAN）；现维护 [kjyv/FloBaRoID](https://github.com/kjyv/FloBaRoID)
- **链接：** <https://github.com/kjyv/FloBaRoID>
- **论文：** Bethge, Malzahn, Tsagarakis, Caldwell, RAAD 2017，DOI <https://doi.org/10.1007/978-3-319-61276-8_18>
- **许可：** LGPL-3.0
- **Stars：** 87（2026-08-13）
- **语言：** Python
- **入库日期：** 2026-08-13
- **一句话说明：** Fourier 激励 + OLS/WLS/SDP 辨识浮动基惯性参数；两步法先用基座 wrench 去摩擦、再拟合关节摩擦；结果写回 URDF。
- **沉淀到 wiki：** [`wiki/entities/flobaroid.md`](../../wiki/entities/flobaroid.md)

## 开源核查（2026-08-13）

| 项 | 结论 |
|----|------|
| GitHub | 非 fork；默认 `master`；2026-07 仍有推送 |
| 许可 | LGPL-3.0 |
| 项目页 | 无独立 Pages（`homepage` 空）；入口即本仓 + `documentation/TUTORIAL.md` |
| 可运行入口 | `uv run gui.py` / `trajectory.py` / `simulator.py` / `identifier.py`；示例 `configs/kuka_lwr4.yaml` |
| 动力学核 | iDynTree（需 eigen/swig）；轨迹优化用 IPOPT |

**结论：已开源。**

## 仓内入口

| 路径 | 角色 |
|------|------|
| `trajectory.py` | Fourier 周期激励，D-最优 + 碰撞约束 |
| `excite.py` | ROS/MoveIt 或 YARP 真机激励并记录 |
| `simulator.py` | 无真机时用 ID 加摩擦/噪声合成测量 |
| `identifier.py` | OLS / WLS / SDP；两步摩擦 |
| `gui.py` | 上述步骤的图形前端 |

## 与仓库内实体的关系

- [FloBaRoID 实体](../../wiki/entities/flobaroid.md)
- [关节执行器参数辨识](../../wiki/methods/joint-actuator-parameter-identification.md)
- [论文簇](../papers/joint_actuator_parameter_identification.md)
- 对照：[BAM](../../wiki/entities/bam-better-actuator-models.md)（摆锤、无力矩传感）、[PACE](../../wiki/entities/paper-pace-sim2real-legged-robots.md)（悬空 chirp）
