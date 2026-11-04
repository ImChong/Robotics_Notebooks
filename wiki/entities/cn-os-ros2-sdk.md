---
type: entity
tags: [repo, china-embodied-opensource, open-source, project, ti5robot]
status: draft
updated: 2026-09-06
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/ros2_sdk.md
  - ../../sources/repos/linkerhand-ros2-sdk.md
summary: "星动纪元 开源项目 ros2_sdk（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - roboterax
---

# ros2_sdk

## 一句话定义

**ros2_sdk** 是 [星动纪元](https://github.com/roboterax) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

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
| 机构 | 星动纪元 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/roboterax |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `ros2_sdk` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 launch simple_trajectory_server simple_trajectory_server.launch.py`（README「4. Run demo」第 1 步；启动 `console_scripts` 入口 `simple_trajectory_server` → `simple_trajectory_server.simple_trajectory_node:main`）；客户端：`ros2 action send_goal /simple_trajectory_action xbot_common_interfaces/action/SimpleTrajectory "traj_type: 0 duration: 4.0"`

```mermaid
sequenceDiagram
    participant Cli as ros2 action 客户端（README 示例）
    participant Launch as simple_trajectory_server.launch.py
    participant Node as MinimalActionServer（trajectory_server）
    participant Js as joint_states / imu 话题（上游）
    participant Ctrl as hybrid_body_controller/commands（下游，本仓库外）
    Cli->>Launch: ros2 launch（加载 config/config.yaml 参数）
    Launch->>Node: 启动 simple_trajectory_node:main
    Node->>Node: __init__：ActionServer(simple_trajectory_action)，订阅 joint_states 与 /imu_broadcaster/imu，声明参数，创建发布器
    Js->>Node: joint_states_cb 写入 self.joint_states
    Cli->>Node: send_goal(traj_type, duration)
    Node->>Node: goal_callback：SIN_WAVE 时检查各关节接近 0，并拒绝已有活动目标；其余 ACCEPT
    Node->>Node: handle_accepted_callback：记录 goal_handle 与 duration，goal_handle.execute()
    Node->>Node: execute_callback 按 traj_type 分派
    alt ZERO：move_to_zero
        loop 直到 end_time（time.sleep 0.01）
            Node->>Node: linear_interpolation 到 target_positions（0）
            Node->>Ctrl: set_joint_positions → publish HybridJointCommand（position / kp / kd）
            Node-->>Cli: publish_feedback(progress)
        end
        Node->>Ctrl: set_joint_positions(end_positions)
        Node->>Node: goal_handle.succeed()
    else SIN_WAVE：sin_wave
        loop 直到 end_time（time.sleep 0.01）
            Node->>Node: start + 0.04 · sin(2π · 0.1 · t)，各关节
            Node->>Ctrl: set_joint_positions → publish HybridJointCommand
            Node-->>Cli: publish_feedback(progress)
        end
        Node->>Node: goal_handle.succeed()
    end
    Note over Cli,Node: 循环内若 goal 非 active 则 abort，若收到 cancel 则 canceled（cancel_callback 恒 ACCEPT）
```

图依据：ros2_sdk@b12bd12，`README.md`、`simple_trajectory_server/launch/simple_trajectory_server.launch.py`、`simple_trajectory_server/setup.py`、`simple_trajectory_server/simple_trajectory_server/simple_trajectory_node.py`、`simple_trajectory_server/config/config.yaml`；未运行代码，未读 rl_controller 与 xbot_common_interfaces 的实现。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [linkerhand-ros2-sdk 源码归档](../../sources/repos/linkerhand-ros2-sdk.md)（<https://github.com/linker-bot/linkerhand-ros2-sdk>）

- [ros2_sdk 源码归档](../../sources/repos/ros2_sdk.md)（<https://github.com/roboterax/ros2_sdk>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：roboterax/ros2_sdk@b12bd12，读过 README.md、simple_trajectory_server/launch/simple_trajectory_server.launch.py、simple_trajectory_server/setup.py、simple_trajectory_server/simple_trajectory_server/simple_trajectory_node.py、simple_trajectory_server/config/config.yaml；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [星动纪元 官方组织](https://github.com/roboterax)
