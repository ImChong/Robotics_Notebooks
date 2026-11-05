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
  - ../../sources/repos/alicia-d-ros2.md
summary: "玄雅科技 开源项目 Alicia-D-ROS2（SDK/驱动）：Alicia-D 机械臂 ROS2 支持包…"
institutions:
  - xuanya
---

# Alicia-D-ROS2

## 一句话定义

**Alicia-D-ROS2** 是 [玄雅科技](https://github.com/Synria-Robotics) 公开的 **SDK/驱动** 开源项目：Alicia-D 机械臂 ROS2 支持包

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
| 机构 | 玄雅科技 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/Synria-Robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Alicia-D-ROS2` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 launch alicia_d_moveit real_robot.launch.py`（MoveIt + ros2_control 真机路径）；另有独立驱动入口 `ros2 launch alicia_d_driver alicia_d_driver.launch.py`（`alicia_d_driver_node`，直接收发 `/joint_states`、`/joint_commands`），图中只画 MoveIt 路径。

```mermaid
sequenceDiagram
    participant L as real_robot.launch.py
    participant CM as ros2_control_node（controller_manager，200 Hz）
    participant HI as AliciaDHardwareInterface（alicia_d_hardware_interface.cpp）
    participant DP as AliciaDDataParserControl（解析线程）
    participant SC as SerialCommunicator（read_thread_loop）
    participant HW as Alicia-D 机械臂（串口）
    participant MG as move_group（MoveItSimpleControllerManager）
    L->>CM: 以 get_versioned_moveit_config 的 robot_description + ros2_controllers.yaml 启动
    L->>CM: spawner：joint_state_broadcaster、Alicia_controller、Gripper_controller
    L->>MG: 启动 move_group、rviz2、robot_state_publisher
    CM->>HI: pluginlib 加载 alicia_d_driver/AliciaDHardwareInterface，on_init / on_configure
    HI->>SC: new SerialCommunicator(port)（port 为空则 find_serial_port）
    CM->>HI: on_activate
    HI->>DP: start_parsing_thread()；acquire_info(version/temperature/velocity/self_check)
    loop 每个控制周期
        CM->>HI: read()
        HI->>DP: 周期性 acquire_info("joint", false)，get_joint_state()
        DP->>SC: get_packet()（读线程校验后入队的帧）
        SC-->>HW: 请求关节数据帧
        HW-->>SC: 帧 [AA][Cmd][Func][Len][Data][CRC][FF]
        HI-->>CM: 更新 hw_positions_state_ / hw_velocities_state_
    end
    MG->>CM: FollowJointTrajectory action 发送给 Alicia_controller / Gripper_controller
    CM->>HI: write()（命令接口 position/velocity）
    HI->>DP: set_joint_and_gripper(joint_angles, gripper_value, speed_deg_s)
    DP->>SC: build_joint_frame() → write_packet(frame)
    SC->>HW: 串口写入关节与夹爪目标
```

图依据：Synria-Robotics/Alicia-D-ROS2@2c35df8，读过 README.md、alicia_d_moveit/launch/real_robot.launch.py、alicia_d_moveit/config/ros2_controllers.yaml、moveit_controllers.yaml、alicia_d_descriptions.ros2_control.xacro、alicia_d_driver/launch/alicia_d_driver.launch.py、src/alicia_d_driver_node.cpp、src/alicia_d_hardware_interface.cpp、src/alicia_d_data_parser_control.cpp、src/serial_communicator.cpp（均为部分阅读）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Alicia-D-ROS2 源码归档](../../sources/repos/alicia-d-ros2.md)（<https://github.com/Synria-Robotics/Alicia-D-ROS2>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Synria-Robotics/Alicia-D-ROS2@2c35df8，读过 README.md、alicia_d_moveit/launch/real_robot.launch.py、alicia_d_moveit/config/ros2_controllers.yaml、alicia_d_moveit/config/moveit_controllers.yaml、alicia_d_descriptions.ros2_control.xacro、alicia_d_driver/launch/alicia_d_driver.launch.py、alicia_d_driver/src/{alicia_d_driver_node,alicia_d_hardware_interface,alicia_d_data_parser_control,serial_communicator}.cpp（后四者为部分阅读）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [玄雅科技 官方组织](https://github.com/Synria-Robotics)
