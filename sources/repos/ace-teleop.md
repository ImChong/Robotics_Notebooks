# ACE Teleop（ACETeleop/ACETeleop）

> 来源归档

- **标题：** ACE Teleop
- **类型：** repo
- **来源：** UC San Diego
- **链接：** <https://github.com/ACETeleop/ACETeleop>
- **硬件仓库：** <https://github.com/ACETeleop/ACE_hardware>
- **项目页：** <https://ace-teleop.github.io/>
- **论文：** <https://arxiv.org/abs/2408.11805>
- **许可：** 未声明标准 SPDX 许可证（GitHub API 为 `NOASSERTION`）
- **入库日期：** 2026-07-28
- **一句话说明：** 公开 ACE 的 `server → controller → simulation/real robot` 软件链、机器人配置、Dynamixel 校准工具及配套 STL 硬件。
- **开源状态：** **源码与硬件文件已公开，许可边界待确认**。
- **沉淀到 wiki：** [`paper-notebook-ace-a-cross-platform-visual-exoskeletons-system.md`](../../wiki/entities/paper-notebook-ace-a-cross-platform-visual-exoskeletons-system.md)

## 仓库概况（2026-07-28）

| 字段 | 值 |
|------|-----|
| 安装 | `pip install -e .` |
| 服务端 | `scripts/start_server.py --config ...` |
| 控制命令 | `scripts/teleop_cmd.py --config ...` |
| 仿真 | `scripts/teleop_sim.py --config ...`（可选 Isaac Gym） |
| 校准 | `ace_teleop/dynamixel/calibration/` |
| 配置 | `ace_teleop/configs/server/` |

README 明确支持 xArm+Ability、Franka+gripper、H1+Inspire、GR-1+gripper；真机控制需要维护者把 `teleop_cmd` 接入具体机器人接口。

## 对 wiki 的映射

- 项目页：[`ace-teleop.md`](../sites/ace-teleop.md)
- 论文来源：[`humanoid_pnb_ace.md`](../papers/humanoid_pnb_ace.md)
- 遥操作路线：[`depth-teleoperation.md`](../../roadmap/depth-teleoperation.md)
