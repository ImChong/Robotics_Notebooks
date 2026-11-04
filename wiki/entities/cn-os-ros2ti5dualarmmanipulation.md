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
  - ../../sources/repos/ros2ti5dualarmmanipulation.md
summary: "钛虎机器人 开源项目 ROS2Ti5DualArmManipulation（SDK/驱动）：人形双臂操作 ROS2 控制接口…"
institutions:
  - ti5robot
---

# ROS2Ti5DualArmManipulation

## 一句话定义

**ROS2Ti5DualArmManipulation** 是 [钛虎机器人](https://github.com/ti5robot) 公开的 **SDK/驱动** 开源项目：人形双臂操作 ROS2 控制接口

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第一层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 钛虎机器人 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/ti5robot |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `ROS2Ti5DualArmManipulation` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 launch arms demo.launch.py`（启动 RViz / MoveIt）后，另开终端执行 `ros2 run arms_demo arms_demo`（`ROS2双臂接口文档.md` 使用步骤 3–4；`arms_demo/src/arms_demo.cpp` 的 `main`）

```mermaid
sequenceDiagram
    participant Main as arms_demo.cpp main
    participant Arm as ArmsDemo（Ti5_Arms.cpp）
    participant MG as MoveGroupInterface arm_L / arm_R（MoveIt2）
    participant Can as libcontrolcan VCI_*（USBCAN 设备）
    participant Rviz as RViz / MoveIt 规划（display_planned_path）
    Main->>Arm: make_shared ArmsDemo(arms_demo, arm_L, arm_R)
    Arm->>MG: 构造两个 MoveGroupInterface，设置容差与速度缩放
    Arm->>Arm: create_subscription(display_planned_path, callback)
    Arm->>Arm: 启动 SingleThreadedExecutor 线程
    Arm->>Can: init_can：VCI_OpenDevice → VCI_InitCAN(ch0 / ch1) → VCI_StartCAN
    Main->>Arm: move_up()
    Arm->>Can: sendCanCommand(7，左臂 ID 23–29，通道 0)：速度 +500，速度 -500，位置 0
    Arm->>Can: sendCanCommand(7，右臂 ID 16–22，通道 1)：同上
    Main->>Main: while rclcpp::ok，loop_rate 500 Hz 循环
    Note over Main: arms_demo.cpp 中 move_L_by_joint / move_L_by_pos / get_* 等调用已注释，未执行
    Rviz->>Arm: 规划结果 DisplayTrajectory（display_planned_path）
    Arm->>Arm: callback：按 joint_names 查 CAN ID，取轨迹末点角度 ×57.3 ×101 ×65536 ÷360
    Arm->>Can: sendCanCommand(7，位置帧，通道 cnt/7)
    Note over Arm,MG: move_L_by_pos 路径：setPoseTarget → plan → 成功则 execute（execute 的硬件侧实现不在本仓库）
```

图依据：Ti5_arms 仓库 @76a5c0d，`ROS2双臂接口文档.md`、`arms_demo/src/arms_demo.cpp`、`Ti5_Arms/src/Ti5_Arms.cpp`（构造函数、init_can、callback、move_up、move_L_by_pos 段）；未运行代码。callback 与 execute 之间的实际执行关系未在源码中完全确认，图中仅画源码可见的调用。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [ROS2Ti5DualArmManipulation 源码归档](../../sources/repos/ros2ti5dualarmmanipulation.md)（<https://github.com/ti5robot/ROS2Ti5DualArmManipulation>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：ti5robot/ROS2Ti5DualArmManipulation@76a5c0d，读过 README.md、ROS2双臂接口文档.md、arms_demo/src/arms_demo.cpp、Ti5_Arms/src/Ti5_Arms.cpp（构造函数、init_can、callback、move_up、move_L_by_pos 段）；另仅抽看 Ti5_py/setup.py 与其函数列表，未读实现；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [钛虎机器人 官方组织](https://github.com/ti5robot)
