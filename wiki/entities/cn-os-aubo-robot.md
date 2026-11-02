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
  - ../../sources/repos/aubo_robot.md
summary: "遨博智能 开源项目 aubo_robot（工程与工具）：为AUBO协作机器人提供ROS工业包、机器人描述、驱动、MoveIt和Gazebo集成。…"
institutions:
  - aubo
---

# aubo_robot

## 一句话定义

**aubo_robot** 是 [遨博智能](https://github.com/AuboRobot) 公开的 **工程与工具** 开源项目：为AUBO协作机器人提供ROS工业包、机器人描述、驱动、MoveIt和Gazebo集成。

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
| 机构 | 遨博智能 |
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/AuboRobot、https://gitee.com/aubo-nxrobo |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `aubo_robot` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`roslaunch aubo_i5_moveit_config moveit_planning_execution.launch sim:=false robot_ip:=<控制器 IP>`（以 i5 为例，各型号 `<robot>_moveit_config` 同构；`sim` 参数在该 launch 内被声明但节点启动并不依赖它）

```mermaid
sequenceDiagram
    participant L as moveit_planning_execution.launch
    participant MG as move_group (MoveIt)
    participant JTA as aubo_joint_trajectory_action (aubo_controller/src/joint_trajectory_action.cpp)
    participant Sim as aubo_robot_simulator (aubo_controller/script/aubo_controller/aubo_robot_simulator, Python)
    participant Drv as aubo_driver (aubo_driver/src/aubo_driver.cpp)
    participant Ctl as AUBO 控制器 (auborobotcontroller 库, robotService* TCP 接口)

    L->>Sim: 启动 aubo_robot_simulator
    L->>JTA: 启动 aubo_joint_trajectory_action（follow_joint_trajectory 动作服务）
    L->>Drv: 启动 aubo_driver，参数 /server_host = robot_ip
    L->>MG: include move_group.launch（controllers.yaml: aubo_i5_controller / FollowJointTrajectory）
    Drv->>Ctl: run() → connectToRobotController：robotServiceLogin（send / receive / mac_size 三路）
    Drv->>Ctl: robotServiceEnterTcp2CanbusMode（切到 ROS 控制，control_option_ = RosMoveIt）
    Drv->>Drv: createTimer(timerCallback)，起 publishWaypointToRobot 线程
    loop 定时 timerCallback
        Drv->>Ctl: robotServiceGetCurrentWaypointInfo
        Drv-->>JTA: 发布 joint_states / feedback_states / robot_status
        Drv-->>Sim: 发布 /aubo_driver/rib_status、/aubo_driver/real_pose
    end
    MG->>JTA: FollowJointTrajectory goal（goalCB）
    JTA->>Sim: 发布 joint_path_command（JointTrajectory）
    Sim->>Sim: trajectory_callback：缩放速度、add_motion_waypoint；_motion_worker 五次多项式插值
    Sim->>Drv: 发布 moveItController_cmd（JointTrajectoryPoint）
    Drv->>Drv: moveItPosCallback 入 buf_queue_，积累后 start_move_
    loop driver_node.cpp main 循环：updateControlStatus（UPDATE_RATE_）
        Drv->>Drv: setRobotJointsByMoveIt：buf_queue_ 出队 → ros_motion_queue_
        Drv->>Ctl: publishWaypointToRobot 线程：robotServiceSetRobotPosData2Canbus
    end
    Note over Drv,Ctl: 急停 / 保护停止时，驱动发布 aubo_driver/cancel_trajectory 并按 otgVelocityMode 减速后清空缓冲
```

图依据：AuboRobot/aubo_robot@32210c0，读过 `README.md`、`aubo_robot/aubo_i5_moveit_config/launch/moveit_planning_execution.launch`、`move_group.launch`、`config/controllers.yaml`、`aubo_robot/aubo_driver/src/aubo_driver.cpp`、`driver_node.cpp`、`aubo_driver/CMakeLists.txt`、`aubo_robot/aubo_controller/CMakeLists.txt`、`aubo_controller/script/aubo_controller/aubo_robot_simulator`、`aubo_controller/src/joint_trajectory_action.cpp`（局部）。`robotService*` 来自预编译库 `auborobotcontroller`，其内部未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [aubo_robot 源码归档](../../sources/repos/aubo_robot.md)（<https://github.com/AuboRobot/aubo_robot>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：AuboRobot/aubo_robot@32210c0，读过 README.md、aubo_i5_moveit_config/launch/moveit_planning_execution.launch 与 move_group.launch、config/controllers.yaml、aubo_driver/src/aubo_driver.cpp 与 driver_node.cpp、aubo_driver/CMakeLists.txt、aubo_controller/CMakeLists.txt、aubo_controller/script/aubo_controller/aubo_robot_simulator、aubo_controller/src/joint_trajectory_action.cpp（局部）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [遨博智能 官方组织](https://github.com/AuboRobot)
