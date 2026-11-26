---
type: method
tags: [robotics, control, architecture]
status: complete
updated: 2026-11-26
summary: "组织任务命令、状态估计、运动控制与执行器接口，并明确各层运行边界。"
sources:
  - ../../sources/blogs/wechat_realxiaoze_robot_simulation_stack_2026-09-20.md
  - ../../sources/sites/platform-method-robot-control-architecture.md
original:
  - ../../sources/sites/platform-method-robot-control-architecture.md
related:
  - ../concepts/whole-body-control.md
  - robot-software-integration.md
  - trajectory-optimization.md
---

# Robot Control System Architecture · 机器人控制系统架构

## 一句话定义

组织任务命令、状态估计、运动控制与执行器接口，并明确各层运行边界。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| WBC | Whole-Body Control | 在全身约束下协调多关节 |
| PD | Proportional–Derivative Control | 依据误差及误差变化率计算输出 |
| DoF | Degree of Freedom | 机构独立运动的自由度 |

## 为什么重要

任务层、伺服与驱动若只约定“发送动作”，出问题时无法区分命令过期、单位不匹配与控制器不稳定。

## 主要技术路线（核心机制）

ros2_control Humble 的 Controller Manager 管理控制器生命周期与硬件接口；实时循环依次读硬件状态、更新控制器、写硬件命令。update_rate 是控制管理器循环参数，不代表视觉、策略或电机内部也是这个频率。

```mermaid
flowchart LR
    N0["上游任务参考"]
    N1["读入硬件状态"]
    N2["更新控制器"]
    N3["写出关节命令"]
    N4["采集下一周期反馈"]
    N0 --> N1 --> N2 --> N3 --> N4
```

## 工程实践

列出关节命令模式、符号、限幅与接口拥有者。记录每个周期的实测间隔与超期次数，另测上游参考龄期、通信延迟。受控台架验证控制器切换、过期命令、传感器失联及已定义的安全停机行为。

## 局限与风险

调度策略只能降低某些抖动；不能证明端到端满足控制截止时间或安全认证。高层规划不得绕过底层安全边界。

## 关联页面

- [全身控制](../concepts/whole-body-control.md)
- [机器人软件系统集成](robot-software-integration.md)
- [轨迹优化](trajectory-optimization.md)

## 参考来源

- [ros2_control Humble 控制管理器与硬件接口](../../sources/sites/platform-method-robot-control-architecture.md)

## 推荐继续阅读

- [Controller Manager](https://control.ros.org/humble/doc/ros2_control/controller_manager/doc/userdoc.html)
- [硬件接口类型](https://control.ros.org/humble/doc/ros2_control/hardware_interface/doc/hardware_interface_types_userdoc.html)
