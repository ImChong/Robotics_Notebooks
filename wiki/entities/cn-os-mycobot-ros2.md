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
  - ../../sources/repos/mycobot_ros2.md
summary: "大象机器人 开源项目 mycobot_ros2（仿真环境）：为myCobot及相关机器人提供ROS2模型、驱动、MoveIt和仿真开发入口。…"
institutions:
  - elephant-robotics
---

# mycobot_ros2

## 一句话定义

**mycobot_ros2** 是 [大象机器人](https://github.com/elephantrobotics) 公开的 **仿真环境** 开源项目：为myCobot及相关机器人提供ROS2模型、驱动、MoveIt和仿真开发入口。

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
| 机构 | 大象机器人 |
| 类别 | 仿真环境 |
| 官方组织 | https://github.com/elephantrobotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `mycobot_ros2` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 launch mycobot_280 slider_control.launch.py`（ROS2 标准命令，取 `mycobot_280` 包 slider 路线；可传 `port:=/dev/ttyUSB0 baud:=115200`）

仓库另有 mecharm、Mybuddy、myArm、mycobot_320 等多套包，本图只取 mycobot_280 一条路线。

```mermaid
sequenceDiagram
    participant U as 用户
    participant LL as slider_control.launch.py
    participant RSP as robot_state_publisher
    participant JG as joint_state_publisher_gui
    participant RV as rviz2
    participant SN as slider_control 节点（Slider_Subscriber）
    participant MC as MyCobot280（pymycobot）
    U->>LL: ros2 launch mycobot_280 slider_control.launch.py
    LL->>LL: robot_description 由 xacro 读取 model 参数指定的 URDF
    LL->>RSP: 启动 robot_state_publisher
    LL->>JG: 启动 joint_state_publisher_gui（gui 参数为 true 时）
    LL->>RV: 启动 rviz2，-d config/mycobot.rviz
    LL->>SN: 启动 slider_control 节点，传入 port 与 baud 参数
    SN->>SN: __init__ 订阅 joint_states（JointState）
    SN->>MC: MyCobot280(port, 115200)，baud 参数已读取但此处写死 115200
    SN->>MC: set_fresh_mode(1)
    U->>JG: 拖动滑条
    JG-->>SN: joint_states 话题（ROS 约定，非本仓库代码）
    SN->>SN: listener_callback 弧度转角度，保留 2 位小数
    SN->>MC: send_angles(data_list, 25)
```

图依据：elephantrobotics/mycobot_ros2@d42ff61，读过 README.md、mycobot_280/mycobot_280/launch/slider_control.launch.py、mycobot_280/mycobot_280/mycobot_280/slider_control.py、mycobot_280/mycobot_280/setup.py（entry_points 段）。pymycobot 库内部未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [mycobot_ros2 源码归档](../../sources/repos/mycobot_ros2.md)（<https://github.com/elephantrobotics/mycobot_ros2>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

- 源码核对（2026-11-02）：elephantrobotics/mycobot_ros2@d42ff61，读过 README.md、mycobot_280/mycobot_280/launch/slider_control.launch.py、mycobot_280/mycobot_280/mycobot_280/slider_control.py、mycobot_280/mycobot_280/setup.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [大象机器人 官方组织](https://github.com/elephantrobotics)
