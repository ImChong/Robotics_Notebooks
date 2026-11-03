---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: draft
updated: 2026-09-06
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/elite_robots_cs_ros2_driver.md
summary: "艾利特机器人 开源项目 Elite_Robots_CS_ROS2_Driver（SDK/驱动）：为艾利特CS系列提供ROS2驱动、标定、机器人描述、控制器、MoveIt和Gazebo仿真集成。…"
institutions:
  - elite-robot
---

# Elite_Robots_CS_ROS2_Driver

## 一句话定义

**Elite_Robots_CS_ROS2_Driver** 是 [艾利特机器人](https://github.com/Elite-Robots) 公开的 **SDK/驱动** 开源项目：为艾利特CS系列提供ROS2驱动、标定、机器人描述、控制器、MoveIt和Gazebo仿真集成。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第五层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 艾利特机器人 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/Elite-Robots |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Elite_Robots_CS_ROS2_Driver` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 源码运行时序图

入口：`ros2 launch eli_cs_robot_driver elite_control.launch.py robot_ip:=... local_ip:=... cs_type:=...`（真机，`use_fake_hardware:=false`）

```mermaid
sequenceDiagram
    participant L as elite_control.launch.py
    participant CM as eli_ros2_control_node (control_node.cpp)
    participant HW as EliteCSPositionHardwareInterface (hardware_interface.cpp)
    participant D as ELITE::EliteDriver / RtsiIOInterface (Elite CS SDK)
    participant CL as eli_components_loader (load_components.cpp)
    participant SP as controller_manager spawner
    participant JTC as scaled_joint_trajectory_controller
    participant R as 艾利特 CS 控制器

    L->>CM: 启动，传入 robot_description + controllers yaml
    L->>CL: 启动，参数 robot_ip
    CL->>CL: 加载 DashboardClient / PrimaryClient / ScriptNode 三个节点并 spin
    L->>SP: spawn joint_state_broadcaster、io_and_status_controller 等
    L->>SP: spawn 初始关节控制器
    CM->>HW: on_init / on_configure
    HW->>D: rtsiInit(robot_ip)，连接 RTSI 并读 output/input recipe
    HW->>D: make_unique EliteDriver(driver_config_)
    D-->>R: 按 driver_config_ 的 script_sender/reverse/trajectory 端口与机器人交互（细节在 SDK 内，未读）
    HW->>HW: on_configure 末尾启动 asyncThread
    loop 控制周期 (cm->read / update / write)
        CM->>HW: read()
        HW->>D: RTSI getActualJointPositions / Velocity / TCPPose ...
        CM->>JTC: update()（按 speed scaling 采样轨迹）
        CM->>HW: write()
        alt 任务 PLAYING 且机器人已连接
            HW->>D: writeServoj / writeSpeedj / writeFreedrive
            D->>R: 经 reverse 端口下发指令
        else
            HW->>D: writeIdle
        end
    end
    Note over HW,D: IO、payload、speed scaling、freedrive、resend_external_script<br/>等异步命令在 write()/updateAsyncIO() 中转发给 EliteDriver / RTSI
```

图依据：Elite-Robots/Elite_Robots_CS_ROS2_Driver@3e9e8ca，读过 `eli_cs_robot_driver/launch/elite_control.launch.py`、`src/control_node.cpp`、`src/hardware_interface.cpp`（on_configure/read/write 部分）、`src/load_components.cpp`、`CMakeLists.txt`、`doc/Usage_CN.md`（启动命令）、`eli_cs_controllers/src/scaled_joint_trajectory_controller.cpp`（头部）。reverse 端口脚本下发的细节在 SDK（见 Elite_Robots_CS_SDK 页），此处未读 SDK 内部。

## 参考来源

- [Elite_Robots_CS_ROS2_Driver 源码归档](../../sources/repos/elite_robots_cs_ros2_driver.md)（<https://github.com/Elite-Robots/Elite_Robots_CS_ROS2_Driver>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Elite-Robots/Elite_Robots_CS_ROS2_Driver@3e9e8ca，读过 eli_cs_robot_driver 的 launch/elite_control.launch.py、src/control_node.cpp、src/hardware_interface.cpp（部分）、src/load_components.cpp、CMakeLists.txt、doc/Usage_CN.md 及 eli_cs_controllers/src/scaled_joint_trajectory_controller.cpp（头部）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [艾利特机器人 官方组织](https://github.com/Elite-Robots)
