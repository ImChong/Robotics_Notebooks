# humanoid-gym-modified

> 来源归档

- **标题：** humanoid-gym-modified
- **类型：** repo / fork
- **来源：** roboman-ly（社区维护）
- **链接：** <https://github.com/roboman-ly/humanoid-gym-modified>
- **上游：** <https://github.com/roboterax/humanoid-gym>
- **Stars：** ~87（2026-06）
- **入库日期：** 2026-06-09
- **一句话说明：** 在官方 **Humanoid-Gym** 上增加开源 **Pandaman** 人形模型与 **Gazebo + ROS Noetic** sim2sim 管线，便于无 MuJoCo 或需 ROS 生态的二次验证。
- **沉淀到 wiki：** [`wiki/entities/humanoid-gym.md`](../../wiki/entities/humanoid-gym.md)（fork 小节）

---

## 相对上游的增量

| 能力 | 官方 humanoid-gym | 本 fork |
|------|-------------------|---------|
| 默认机器人 | RobotEra XBot | + **Pandaman** 开源模型 |
| Sim2Sim | MuJoCo（`sim2sim.py`） | MuJoCo **+ Gazebo**（`humanoid_ros`） |
| 任务示例 | `humanoid_ppo` | `pandaman_ppo` 等 |
| ROS | 无 | **ros-noetic** + `gazebo_ros_control` 等 |

## 依赖栈（增量）

- 与上游相同的 **Python 3.8 / PyTorch 1.13 / Isaac Gym Preview 4**
- **ROS Noetic**（Ubuntu 20.04）：`ros-noetic-gazebo-ros-control`、`joint_trajectory_controller`、`effort_controllers` 等
- `humanoid_ros`：`catkin_make` 构建

## 典型命令（README）

```bash
# Pandaman 训练
python scripts/train.py --task=pandaman_ppo --run_name v1 --headless --num_envs 4096

# Gazebo sim2sim（双终端）
# T1: conda deactivate && roslaunch gazebo_sim robotdiscription_gazebo.launch
# T2: conda activate humanoid && python humanoid_ros/src/gazebo_sim/scripts/sim.py
```

键盘遥操作：`W/S` 前后、`A/D` 横移、`Q/E` 转向。

## 与本仓库其他资料的关系

| 资料 | 关系 |
|------|------|
| [humanoid-gym.md](humanoid-gym.md) | 上游官方仓库 |
| [humanoid_gym_arxiv_2404_05695.md](../papers/humanoid_gym_arxiv_2404_05695.md) | 方法论仍遵循 Humanoid-Gym 论文 |
| Pandaman | 知乎开源人形模型（README 外链）；低成本硬件复现入口 |

## 为何值得保留

- **硬件多样性：** 除 XBot 外提供 **Pandaman** 资产配置与训练任务，适合国内开源人形生态复现。
- **Sim2Sim 路径扩展：** Gazebo 验证补充 MuJoCo，对接 **ROS 部署链** 与仿真互操作习惯不同的团队。
- **维护预期：** 社区 fork，版本可能滞后上游；新功能（如 DWL）以官方仓库为准。
