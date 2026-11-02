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
  - ../../sources/repos/ros2_rm_robot.md
summary: "睿尔曼智能 开源项目 ros2_rm_robot（工程与工具）：提供睿尔曼机械臂ROS2驱动与消息接口，支持规划、感知和双臂系统集成。…"
institutions:
  - realman
---

# ros2_rm_robot

## 一句话定义

**ros2_rm_robot** 是 [睿尔曼智能](https://github.com/RealManRobot) 公开的 **工程与工具** 开源项目：提供睿尔曼机械臂ROS2驱动与消息接口，支持规划、感知和双臂系统集成。

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
| 机构 | 睿尔曼智能 |
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/RealManRobot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `ros2_rm_robot` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 launch rm_bringup rm_bringup.launch.py arm_type:=65-6f`（真机，`mode` 默认 `real`；`mode:=gazebo` 走 rm_gazebo，不连控制器）。下图为真机路径。

```mermaid
sequenceDiagram
    participant B as rm_bringup.launch.py<br/>(_compose_bringup)
    participant D as rm_driver 节点<br/>(RmArm + UdpPublisherNode)
    participant API as 预编译 Rm_Api 库<br/>(rm_init/rm_create_robot_arm)
    participant CT as 机械臂控制器<br/>(arm_ip:tcp_port)
    participant RC as rm_control 节点<br/>(Rm_Control)
    participant MV as MoveIt move_group
    B->>D: include rm_driver 的 launch(config/*.yaml 提供 arm_ip/tcp_port/udp_*)
    D->>API: Arm_Start(): rm_init(RM_TRIPLE_MODE_E)
    D->>API: rm_create_robot_arm(tcp_ip, tcp_port)
    API-->>CT: 建立 TCP 连接(库内部实现未读)
    opt 需要判定控制器代际时
        B->>D: 发布 rm_driver/get_robot_info_cmd
        D-->>B: 订阅 get_robot_info_result 得 robot_controller_version
    end
    B->>B: include rm_description / rm_control / MoveIt launch
    loop udp_timer_callback
        D->>API: rm_realtime_arm_state_call_back
        API-->>D: Udp_Robot_Status_Callback(关节状态)
        D-->>MV: 发布 joint_states 等话题
    end
    MV->>RC: FollowJointTrajectory goal(rm_group_controller/follow_joint_trajectory)
    Note over RC: execute_move: 路点数大于 3 时三次样条插值
    loop timer_callback 每个 tick 一个插值点
        RC->>D: 发布 rm_driver/movej_canfd_cmd(Jointpos, follow)
        D->>API: rm_movej_canfd(...)
        API-->>CT: 透传关节角度(库内部实现未读)
    end
    RC-->>MV: goal succeed
```

图依据：RealManRobot/ros2_rm_robot@711abed，读过 README.md、rm_bringup/launch/rm_bringup.launch.py、rm_driver/src/rm_driver.cpp（Arm_Start、UdpPublisherNode、订阅注册、main）、rm_driver/config/rm_65_config.yaml、rm_control/src/rm_control.cpp、rm_moveit2_config/rm_63_config/config/moveit_controllers.yaml；预编译 Rm_Api 库内部、rm_moveit2 与 rm_gazebo 未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [ros2_rm_robot 源码归档](../../sources/repos/ros2_rm_robot.md)（<https://github.com/RealManRobot/ros2_rm_robot>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：RealManRobot/ros2_rm_robot@711abed，读过 README.md、rm_bringup/launch/rm_bringup.launch.py、rm_driver/src/rm_driver.cpp 与 config/rm_65_config.yaml、rm_control/src/rm_control.cpp、rm_moveit2_config/rm_63_config/config/moveit_controllers.yaml；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [睿尔曼智能 官方组织](https://github.com/RealManRobot)
