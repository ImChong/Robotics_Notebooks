# CHILD（uiuckimlab/CHILD）

> 来源归档

- **标题：** CHILD Whole-Body Humanoid Teleoperation
- **类型：** repo
- **来源：** UIUC KIMLAB
- **链接：** <https://github.com/uiuckimlab/CHILD>
- **项目页：** <https://uiuckimlab.github.io/CHILD-pages/>
- **论文：** <https://arxiv.org/abs/2508.00162>
- **许可：** 仓库未见许可证声明
- **入库日期：** 2026-07-28
- **一句话说明：** 公开 CHILD 3D 打印硬件、BOM、ROS 2 leader 接口、G1 全身/上身控制脚本与完整启动说明。
- **开源状态：** **源码/设计公开，许可边界未明确**。
- **沉淀到 wiki：** [`paper-notebook-child-a-whole-body-humanoid-teleoperation-system.md`](../../wiki/entities/paper-notebook-child-a-whole-body-humanoid-teleoperation-system.md)

## 仓库概况（2026-07-28）

| 字段 | 值 |
|------|-----|
| 硬件 | `hardware/BOM.md` 与 3D 打印件 |
| leader 接口 | `hw_interface/teleop_leaders` |
| follower 软件 | `teleop_sw/` |
| CHILD 启动 | `ros2 launch teleop_leaders leader_hw_g1_all_limbs.launch.py` |
| G1 上身 | `python -m run_g1_upper_body` |
| G1 全身 | `python -m run_g1_full_body_teleop` |

复现还依赖 PAPRAS-V0-Public、DynamixelSDK、Unitree SDK、ROS 2 Humble 与特定 G1 网络/安全模式。

## 对 wiki 的映射

- 项目页：[`child-teleoperation.md`](../sites/child-teleoperation.md)
- 论文来源：[`humanoid_pnb_child.md`](../papers/humanoid_pnb_child.md)
- 遥操作路线：[`depth-teleoperation.md`](../../roadmap/depth-teleoperation.md)
