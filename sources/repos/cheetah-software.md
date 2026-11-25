# Cheetah-Software（MIT Biomimetics）

> 来源归档（repo）

- **标题：** Cheetah-Software
- **类型：** repo
- **来源：** MIT Biomimetic Robotics Lab
- **链接：** https://github.com/mit-biomimetics/Cheetah-Software
- **入库日期：** 2026-07-25（2026-11-07 文档通读；2026-11-21 通过 GitHub API 补读 `robot/src/` 与 `user/MIT_Controller/` 核心控制环源码入口）
- **核对范围：** 已读完 `master` 分支 `README.md`、`documentation/getting_started.md`、`documentation/running_mini_cheetah.md`，并通读执行入口与主循环源码：`user/MIT_Controller/main.cpp` → `robot/src/main_helper.cpp`（参数解析与 `SimulationBridge` / `MiniCheetahHardwareBridge` 派发）→ `robot/src/RobotRunner.cpp`（`init()` 构建 MiniCheetah 动力学模型、状态估计与 `LegController`；`run()` 驱动状态估计 + 控制器调用）→ `robot/src/SimulationBridge.cpp`（共享内存与控制周期握手）→ `robot/src/HardwareBridge.cpp`（`init prefault`、RT 调度、LCM 订阅线程）→ `user/MIT_Controller/MIT_Controller.cpp`（`initializeController()` 与 `runController()` 步态调度/FSM）。
- **通读：** 已完成
- **一句话说明：** MIT Mini Cheetah / Cheetah 3 开源机器人与仿真控制软件：含 Convex MPC、WBIC 等经典控制栈入口，是复现 Mini Cheetah 模型基 locomotion 的社区事实标准仓库。
- **开源状态：** **已开源**（公共 GitHub；约 3k+ stars）
- **沉淀到 wiki：** [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)、[paper-wbic-mpc-mini-cheetah](../../wiki/entities/paper-wbic-mpc-mini-cheetah.md)

---

## 文档核对要点（2026-11-07）

- **目录：** `common`（动力学库与工具）、`resources`（可视化用 CAD 等）、`robot`（机器人程序）、`sim`（仿真，唯一依赖 Qt 的程序）、`third-party`（改动过的小型第三方库）。
- **依赖：** Qt 5.10+（需 gamepad 库）、LCM 1.3.1、Eigen、Ipopt 等；构建需先运行 `make_types.sh` 生成 LCM 类型；`common/test-common` 中 OSQP（求解器非确定性）与 CASADI（运行时找库）两项测试偶尔失败，文档称可忽略后者。
- **仿真：** 运行 `sim/sim`，选 Mini Cheetah + Simulator；另开终端运行 `user/MIT_Controller/mit_ctrl m s`（`3`=Cheetah 3、`m`=Mini Cheetah；`s`=仿真、`r`=真机），通过共享内存与仿真通信；用 Logitech F310 手柄驱动。
- **自写控制器：** 在 `user/` 下新建文件夹并加入 `CMakeLists.txt`；`runController` 以 **1 kHz** 被调用；`_legController` 数据与硬件同步约 **700 Hz**，腿指令（前馈力矩、前馈力、笛卡尔 / 关节 PD）会相加。
- **真机：** `cmake -DMINI_CHEETAH_BUILD=TRUE`，`send_to_mini_cheetah.sh` 拷贝，ssh 登录机载电脑（开机约 2 分钟），配置 LCM 网络后运行 `mc_run.sh` / `run_mc.sh`；文档注明当时真机上**不支持**完整状态估计（位置、速度）与完整可视化数据，只有姿态估计。
- **通信：** 用 LCM（非 ROS）连接控制接口与真机，也用于仿真调试。

## 核心定位

为 Mini Cheetah 硬件与仿真提供统一软件栈；大量后续视觉/学习论文在其上叠加感知与策略层。

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md) | 平台 |
| [paper-mini-cheetah-platform](../../wiki/entities/paper-mini-cheetah-platform.md) | 平台论文 |
| [paper-wbic-mpc-mini-cheetah](../../wiki/entities/paper-wbic-mpc-mini-cheetah.md) | WBIC+MPC |
| [bgkatz](./bgkatz.md) | 硬件/驱动侧公开仓库 |

## 社区学习入口（非官方）

| 仓库 | 说明 |
|------|------|
| [quadruped_ctrl](./derek_th_wang_quadruped_ctrl.md) | PyBullet + ROS 仿真控制 |
| [CHAMP](./chvmp_champ.md) | ROS 四足框架，借鉴 Cheetah 思想 |
| [mini_cheetah_ROS](./gleboss1_mini_cheetah_ros.md) | 社区 ROS 移植（星标低，仅入门） |

亦见策展：[mit_mini_cheetah_learning_stack_curator](../personal/mit_mini_cheetah_learning_stack_curator.md)。

