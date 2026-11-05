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
  - ../../sources/repos/wujihandros2.md
summary: "舞肌科技 开源项目 wujihandros2（工程与工具）：为Wuji Hand提供ROS2状态发布与实时控制接口，可将灵巧手接入遥操作、规划和机器人学习系统。…"
institutions:
  - wuji-robotics
---

# wujihandros2

## 一句话定义

**wujihandros2** 是 [舞肌科技](https://github.com/wuji-technology) 公开的 **工程与工具** 开源项目：为Wuji Hand提供ROS2状态发布与实时控制接口，可将灵巧手接入遥操作、规划和机器人学习系统。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第三层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 舞肌科技 |
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/wuji-technology |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `wujihandros2` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 launch wujihand_bringup wujihand.launch.py`（可加 `hand_name:=hand_0`、`serial_number:=<SN>` 或 `hand_side:=left|right`、`rviz:=true`；驱动节点为 `wujihand_driver_node`，入口 `wujihand_driver/src/main.cpp`）。

```mermaid
sequenceDiagram
    autonumber
    participant L as wujihand_bringup/launch/wujihand.launch.py
    participant N as wujihand_driver_node.cpp WujiHandDriverNode
    participant HW as wujihandcpp device/hand.hpp Hand 与 controller
    participant AP as ROS2 上层节点 或 ros2 命令行
    participant RSP as wujihand_bringup/launch/common.py robot_state_publisher

    L->>N: 启动 wujihand_driver_node 传入 serial_number hand_side publish_rate filter_cutoff_freq diagnostics_rate
    N->>N: declare_parameter 并读取参数
    N->>HW: connect_hardware 按 serial_number 或 hand_side 构造 Hand
    N->>HW: realtime_controller true 与 LowPass 滤波器
    N->>HW: write Enabled true 使能关节
    N->>HW: read Handedness 与固件版本
    N->>HW: 读取 ActualPosition 并发送初始目标
    N->>N: 创建 joint_states 与 hand_diagnostics 发布者
    N->>N: 创建 joint_commands 订阅 以及 set_enabled reset_error 服务
    N->>N: 创建 state 定时器 默认 1000 Hz 与 diagnostics 定时器 默认 10 Hz
    L->>RSP: TimerAction 2 秒后 spawn_robot_state_publisher
    loop state 定时器
        N->>HW: 读 controller 的 actual position 与 effort
        N->>AP: 发布 joint_states
    end
    loop diagnostics 定时器
        N->>HW: 读温度 电压 关节温度 错误码 效力限制
        N->>AP: 发布 hand_diagnostics
    end
    AP->>N: joint_commands 发送 JointState
    N->>N: command_callback 按名字或位置顺序组装 5x4 目标
    N->>HW: controller set_joint_target_position
    AP->>N: set_enabled 服务请求
    N->>HW: write Enabled 作用于全部或指定关节
    N-->>AP: SetEnabled 响应
    Note over N,HW: 节点析构时 disconnect_hardware 失能关节
```

图依据：wuji-technology/wujihandros2@95000ed，实际读过 `README.md`、`wujihand_bringup/launch/wujihand.launch.py`（前半段）、`wujihand_driver/src/main.cpp`、`wujihand_driver/src/wujihand_driver_node.cpp`（构造、`connect_hardware`、`command_callback`、`publish_state`、`set_enabled_callback` 等关键段）。`common.py`、`publish_diagnostics` 细节与 `wujihandcpp` SDK 内部未读，`robot_state_publisher` 仅据 launch 调用名画图。未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [wujihandros2 源码归档](../../sources/repos/wujihandros2.md)（<https://github.com/wuji-technology/wujihandros2>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：wuji-technology/wujihandros2@95000ed，读过 README.md、wujihand_bringup/launch/wujihand.launch.py、wujihand_driver/src/main.cpp、wujihand_driver/src/wujihand_driver_node.cpp；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [舞肌科技 官方组织](https://github.com/wuji-technology)
