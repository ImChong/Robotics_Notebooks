# ros2smolvla_docker（ROS2SmolVLA 容器入口）

> 来源归档

- **标题：** ros2smolvla_docker
- **类型：** repo
- **来源：** University of Augsburg / una-auxme（Chair of Mechatronics）
- **链接：** <https://github.com/una-auxme/ros2smolvla_docker>
- **项目页：** <https://una-auxme.github.io/en/projects/ros2smolvla/>
- **论文：** [arXiv:2608.23320](https://arxiv.org/abs/2608.23320) — 归档见 [`sources/papers/ros2smolvla_arxiv_2608_23320.md`](../papers/ros2smolvla_arxiv_2608_23320.md)
- **许可：** Apache-2.0
- **入库日期：** 2026-08-26
- **一句话说明：** ROS2SmolVLA 官方推荐入口：三容器（真机驱动 / Gazebo 仿真 / LeRobot）把 SmolVLA 接到 UR 协作臂；README 给出 `lerobot-record` / `lerobot-train` 与笛卡尔速度控制环。
- **沉淀到 wiki：** [`wiki/entities/paper-ros2smolvla.md`](../../wiki/entities/paper-ros2smolvla.md)

---

## 核心定位

在 **ROS 2 Jazzy + Ubuntu 24.04** 上，用 Docker 隔离 CUDA / PyTorch / ROS 2，让 Hugging Face [LeRobot](https://huggingface.co/lerobot) 的 SmolVLA 对 **Universal Robots** 家族发笛卡尔速度命令。默认硬件叙事是 **UR10e + Robotiq Hand-E**。

## 仓库结构（截至 2026-08-26）

| 路径 | 作用 |
|------|------|
| `docker-compose.yaml` / `docker-compose.gpu.yaml` | `sim` / `real` profile；GPU 叠加文件 |
| `docker/` | 三容器定义 |
| `src/` | 组包源码 |
| `smolvla_ur10e_lerobot.repos` 等 | vcs 依赖清单（LeRobot / 真机 / 仿真） |
| `readme.md` | 采集–训练–推理工作流 |

## 姊妹仓（同组织，均 Apache-2.0）

| 仓库 | 说明 |
|------|------|
| [ros2smolvla_interface_lerobot](https://github.com/una-auxme/ros2smolvla_interface_lerobot) | LeRobot↔ROS 2 经纪（fork 自 `ycheng517/lerobot-ros`） |
| [ros2smolvla_interface_camera](https://github.com/una-auxme/ros2smolvla_interface_camera) | ROS 图像 topic → LeRobot 相机 |
| [ros2smolvla_ur10e_real](https://github.com/una-auxme/ros2smolvla_ur10e_real) | 真机 bringup |
| [ros2smolvla_ur10e_sim](https://github.com/una-auxme/ros2smolvla_ur10e_sim) | Gazebo 数字孪生 |

README 另指向 Cartesian Controllers（FZI）与 Robotiq Hand-E driver（AGH-CEAI）。

## 运行入口（README）

```bash
docker compose --profile sim --profile real build
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml --profile real up
# 容器内
ros2 launch ros2smolvla_ur10e_real ur.launch.py
lerobot-record --robot.type=ur_10e_real --policy.path=<HF_USER>/<MODEL> ...
```

- **动作：** 笛卡尔速度 \((\delta x,\delta y,\delta z,\delta r,\delta p,\delta y)\) → `/servo_node/delta_twist_commands`，再经 `robot_cartesian_operator` 写成 `/cartesian_motion_controller/target_pose`。
- **夹爪：** 过开合中点才发 action goal，避免每周期开关。
- **网络：** 真机默认 `192.168.56.102`，主机 `192.168.56.101`；建议 10 Gbit 传相机流。
- **HF 权重：** `una-auxme/ROS2SmolVLA_ur10e_no_joints_crop_pick_place`（顶视裁 720×720；观测仅末端笛卡尔位姿；动作 6 维速度 + gripper）。

## 开源边界

**已开源、可运行。** 入口仓含 compose、repos 清单与工作流；权重与 349 episode 数据在 Hugging Face。仿真仓已发布，但论文写明仿真数据未用于正式验证。

## 对 wiki 的映射

- 实体页：[ROS2SmolVLA](../../wiki/entities/paper-ros2smolvla.md)
- 方法页：[VLA](../../wiki/methods/vla.md)
- 实体页：[LeRobot](../../wiki/entities/lerobot.md)
