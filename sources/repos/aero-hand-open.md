# Aero Hand Open GitHub 仓库

> 来源归档（ingest）

- **项目名称：** aero-hand-open
- **GitHub 地址：** <https://github.com/TetherIA/aero-hand-open>
- **核心功能：** 开源腱驱拟人手硬件 + MuJoCo 仿真 + mujoco_playground RL + SDK/ROS2 部署栈。
- **入库日期：** 2026-09-01

## 仓库结构

| 路径 | 作用 |
|------|------|
| `hardware/CAD/`、`hardware/Assembly/` | STEP、打印件、BOM |
| `hardware/PCB/` | KiCad / Gerber |
| `sdk/` | Python SDK 与示例序列 |
| `ros2/` | ROS2 接口与策略部署示例 |
| `sim_rl/simulation/` | MuJoCo 手模型（Menagerie 兼容叙事） |
| `sim_rl/mujoco_playground/` | RL 训练环境（域随机化、奖励） |

## 关键复现路径

1. 按 `hardware/Assembly` + [docs.tetheria.ai](https://docs.tetheria.ai/docs/assembly) 组装
2. `sdk/README.md` 配置串口 → `python sdk/examples/run_sequence.py`
3. RL：`sim_rl/mujoco_playground/` 训练 → `ros2/` 零样本部署

## 关联 Wiki 页面

- [Aero Hand Open 论文实体](../../wiki/entities/paper-aero-hand-open.md)
- [Manipulation 任务](../../wiki/tasks/manipulation.md)
