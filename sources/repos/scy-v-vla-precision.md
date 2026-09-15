# VLA-Precision（scy-v/VLA-Precision）

- **URL：** <https://github.com/scy-v/VLA-Precision>
- **组织：** scy-v（USTC）
- **许可证：** Apache-2.0
- **关联论文：** [vla_precision_arxiv_2609_04355](../papers/vla_precision_arxiv_2609_04355.md)
- **项目页：** <https://vla-precision.github.io/>
- **实体页：** [paper-vla-precision](../../wiki/entities/paper-vla-precision.md)

## 一句话说明

VLA-Precision 官方栈：`uv sync --frozen --group stage1|stage2|real-robot`；`main.py --mode norm-stats|train|preprocess|serve-robot|robot-agent-bridge|evaluate` 覆盖 OpenPI 全参微调与 ACoB 在线后训练。Stage II 在线 RL 需 **四进程**：GPU server 上 **learner + actor**，真机侧 **serve-robot + robot-agent-bridge**（同 task/deployment yaml）。配置：`configs/stage1/`、`configs/stage2/tasks/`、`configs/stage2/deployments/`。

### 遥操作与采集（LeRobot 格式）

| 机器人 | 方式 | 仓库 |
|--------|------|------|
| UR5e/UR7e | 键盘 | [scy-v/lerobot_ur5e_keyteleop](https://github.com/scy-v/lerobot_ur5e_keyteleop) |
| UR5e/UR7e | 同构主从 | [scy-v/lerobot_ur5e_isoteleop](https://github.com/scy-v/lerobot_ur5e_isoteleop) |
| 双 UR | VR | [scy-v/lerobot_ur_dual_vrteleop](https://github.com/scy-v/lerobot_ur_dual_vrteleop) |
| Franka | 3D 鼠标/VR 或键盘 | [Shenzhaolong1330/lerobot_franka_teleop](https://github.com/Shenzhaolong1330/lerobot_franka_teleop) |

## 交叉链接

- [VLA-Precision 论文实体](../../wiki/entities/paper-vla-precision.md)
