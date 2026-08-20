# grove-g1

> 来源归档（以 [Adyansh04/grove-g1](https://github.com/Adyansh04/grove-g1) README 与仓库结构为准；截至 2026-08-20）

- **标题：** Grove-G1
- **类型：** repo
- **来源：** Adyansh04（个人维护）
- **链接：** https://github.com/Adyansh04/grove-g1
- **许可：** BSD-3-Clause
- **星标（截至 2026-08-20）：** ~47
- **主要语言：** C++
- **分类：** 人形自主导航 / 操作 / ROS 2 全栈
- **入库日期：** 2026-08-20
- **一句话说明：** 面向 Unitree G1 的 ROS 2 Humble 自主栈：仿真优先（`unitree_mujoco` 同构 DDS）、SLAM Toolbox + Nav2 建图导航、MoveIt 双臂/ Dex3-1 规划、BehaviorTree 编排端到端 pick-and-place 任务。
- **沉淀到 wiki：** 是 → [`wiki/entities/grove-g1.md`](../../wiki/entities/grove-g1.md)
- **相关：** [unitree_mujoco](unitree_mujoco.md)、[unitree_ros2](unitree_ros2.md)、[unitree.md](unitree.md)

---

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 源码 | **已开源、可运行** — `Adyansh04/grove-g1`（BSD-3-Clause；Docker devcontainer + `colcon build`；CI 跑无仿真器测试） |
| 项目页性质 | GitHub 仓库即主入口；无独立 `*.github.io` |
| 数据/权重 | 不适用（系统集成栈；facility 世界地图已提交） |
| 边界 | **尚无真机目标检测管线**；物体位姿在仿真中由专用 source 提供，README 写明硬件上该 source 拒绝启动；**非结构化场景学习型操作** 为下一里程碑、尚未实现 |

---

## README 要点（编译自上游）

### 定位与设计原则

- **平台：** [Unitree G1](https://www.unitree.com/g1) 人形；**ROS 2 Humble** + **CycloneDDS**（pinned loopback；真机改 `GROVE_G1_*` 环境变量即可）。
- **仿真优先：** 对 `unitree_mujoco` 开发；仿真器与真机共用 **相同 DDS topic**，桥接层、导航栈与控制权限逻辑 **无需改代码** 即可上硬件（仅 domain ID / 网卡切换）。
- **两条硬规则（仿真即养成习惯）：**
  1. 每个低层通道 **只有一个发布者** — 控制模式所有权显式。
  2. 臂与行走经 **`rt/arm_sdk`** 权重混合，由机载平衡控制器托腿；直接发 `/lowcmd` 等于自管平衡。

### 当前能力（截至上游 README）

| 能力 | 实现 |
|------|------|
| 建图 / 定位 / 导航 | SLAM Toolbox → AMCL → **Nav2** 到目标位姿 |
| 臂轨迹 | **ros2_control** → `g1_hardware_interface` → `rt/arm_sdk` |
| 手 | Dex3-1 经 `g1_hand_interface`（独立 DDS） |
| 臂规划 | **MoveIt**（单/双臂 + 双手规划组；LiDAR **octomap** 碰撞） |
| 技能 | pick/place **action**；**BehaviorTree.CPP** 编排导航+操作（Groot2 可编辑） |
| 端到端任务 | facility 世界：导航到工作台 → 闭环走近 → 抓取 → 搬运 → 放置 |

### 包结构（`workspace/src/`）

| 包 | 职责 |
|----|------|
| `g1_bringup` | 入口 launch、场景与配置 |
| `g1_description` | G1 URDF + `ros2_control` xacro |
| `g1_hardware_interface` | 14 臂关节 → `rt/arm_sdk` |
| `g1_hand_interface` | 单 Dex3-1 `ros2_control` 插件 |
| `g1_locomotion` | LocoClient 桥、步态整形、行走权限 |
| `g1_motion_service_sim` | 仿真用机载 motion service 替身 |
| `g1_manipulation` | pick/place action + 物体位姿源 |
| `g1_moveit_config` | MoveIt 配置（octomap） |
| `g1_navigation` | SLAM / AMCL / Nav2 |
| `g1_orchestration` | 行为树任务 |
| `g1_sensor_relay` | 仿真内 LiDAR/深度帧 |
| `g1_state_estimation` | `odom`→`base_footprint` TF |

### 开发与部署

- **Host：** Ubuntu 24.04；**容器：** Ubuntu 22.04 + ROS 2 Humble（Unitree SDK 兼容组合）。
- `vcstool` 导入 `workspace.repos` 第三方依赖；`./scripts/manage.sh start|exec` 进 devcontainer。
- 真机切换：`GROVE_G1_CYCLONEDDS_URI`、`GROVE_G1_ROBOT_NIC`、`GROVE_G1_ROS_DOMAIN_ID`；硬件上 LiDAR 前端换 `livox_ros_driver2`，仿真 motion service 换厂商 onboard service。

### 测试与 CI

- `colcon test`：无仿真器套件在 GitHub Actions 跑；含仿真器的测试需本地串行、先 `./scripts/clean-stack.sh`。
- 标签 `simulator` 区分需 MuJoCo 的套件。

---

## 对 wiki 的映射

- 实体：[grove-g1](../../wiki/entities/grove-g1.md)
- 交叉：[unitree-g1](../../wiki/entities/unitree-g1.md)、[unitree-g1-software-stack](../../wiki/entities/unitree-g1-software-stack.md)、[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)、[moveit2](../../wiki/entities/moveit2.md)、[navigation2](../../wiki/entities/navigation2.md)、[slam-toolbox](../../wiki/entities/slam-toolbox.md)
- 四足对照：[autonomy-stack-go2](../../wiki/entities/autonomy-stack-go2.md) — GO2 几何全栈，非人形 loco-manip
