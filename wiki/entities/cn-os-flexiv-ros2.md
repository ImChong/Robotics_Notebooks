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
  - ../../sources/repos/flexiv_ros2.md
summary: "非夕科技 开源项目 flexiv_ros2（工程与工具）：把Flexiv RDK接入ROS2，为感知、规划和控制应用提供标准中间件接口。…"
institutions:
  - flexiv
---

# flexiv_ros2

## 一句话定义

**flexiv_ros2** 是 [非夕科技](https://github.com/flexivrobotics) 公开的 **工程与工具** 开源项目：把Flexiv RDK接入ROS2，为感知、规划和控制应用提供标准中间件接口。

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
| 机构 | 非夕科技 |
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/flexivrobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `flexiv_ros2` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 launch flexiv_bringup flexiv.launch.py robot_sn:=Enlight-L-123456 robot_type:=Enlight-L`（使用 `use_fake_hardware:=true` 时改由 FakeSystem 替代真机硬件）

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户 终端
    participant LF as flexiv_bringup flexiv.launch.py
    participant CM as controller_manager ros2_control_node
    participant HW as flexiv_hardware FlexivHardwareInterface
    participant RDK as flexiv::rdk::Robot 闭源库经网络连接控制器
    participant JS as joint_state_broadcaster
    participant RS as flexiv_robot_states_broadcaster
    participant GR as flexiv_gripper 节点 仅 load_gripper
    participant AC as flexiv_arm_controller JointTrajectoryController
    U->>LF: ros2 launch flexiv_bringup flexiv.launch.py robot_sn robot_type
    LF->>LF: xacro flexiv.urdf.xacro 生成 robot_description (ros2_control:=true)
    LF->>CM: 启动 ros2_control_node, 加载 flexiv_controllers.yaml 与 robot_sn, rdk_control_mode
    CM->>HW: 加载插件 flexiv_hardware/FlexivHardwareInterface
    HW->>RDK: on_init: make_unique Robot(robot_sn)
    LF->>JS: spawner 启动 joint_state_broadcaster
    JS->>CM: 注册并激活关节状态广播
    LF->>RS: spawner 启动 flexiv_robot_states_broadcaster
    alt load_gripper=true
        LF->>GR: OnProcessExit 后启动 flexiv_gripper 节点与 wait_for_gripper_ready
        GR-->>LF: wait_for_gripper_ready 退出码为 0 则继续, 否则关闭系统
    end
    LF->>AC: spawner 启动 flexiv_arm_controller (无夹爪时紧跟 joint_state_broadcaster 退出)
    CM->>HW: on_activate
    HW->>RDK: fault() 若有故障则 ClearFault()
    HW->>RDK: info(), ServoOn(), 循环直到 operational()
    loop 控制循环
        CM->>HW: read()
        HW->>RDK: states(), digital_inputs()
        HW-->>CM: 写入 q, dtheta, tau 等 state interface
        CM->>AC: update() 计算关节命令
        CM->>HW: write()
        alt 模式为 RT_JOINT_POSITION
            HW->>RDK: StreamJointPosition(rt_joint_position_cmds_)
        else 模式为 RT_JOINT_TORQUE
            HW->>RDK: StreamJointTorque(rt_joint_torque_cmds_)
        end
        CM->>RS: update() 发布 flexiv_robot_states 等话题
    end
    CM->>HW: on_deactivate
    HW->>RDK: Stop()
```

图依据：flexivrobotics/flexiv_ros2@a165e88，README.md、flexiv_bringup/launch/flexiv.launch.py、flexiv_bringup/config/flexiv_controllers.yaml（节选）、flexiv_hardware/src/flexiv_hardware_interface.cpp（节选）。RDK 内部闭源，ros2_control 的 read/write 调度为框架行为，未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [flexiv_ros2 源码归档](../../sources/repos/flexiv_ros2.md)（<https://github.com/flexivrobotics/flexiv_ros2>）

- 源码核对（2026-11-02）：flexivrobotics/flexiv_ros2@a165e88，读过 README.md、flexiv_bringup/launch/flexiv.launch.py、flexiv_bringup/config/flexiv_controllers.yaml（节选）、flexiv_hardware/src/flexiv_hardware_interface.cpp（节选）；时序图依据以上文件，未运行代码。

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [非夕科技 官方组织](https://github.com/flexivrobotics)
