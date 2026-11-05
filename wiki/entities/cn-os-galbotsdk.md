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
  - ../../sources/repos/galbotsdk.md
summary: "银河通用 开源项目 GalbotSDK（SDK/驱动）：官方 SDK：真机控制与状态读取的统一接入层…"
institutions:
  - galbot
---

# GalbotSDK

## 一句话定义

**GalbotSDK** 是 [银河通用](https://github.com/GalaxyGeneralRobotics) 公开的 **SDK/驱动** 开源项目：官方 SDK：真机控制与状态读取的统一接入层

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第二层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 银河通用 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/GalaxyGeneralRobotics、https://github.com/PKU-EPIC |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `GalbotSDK` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python examples/g1/python/tutorials/example6_pick_and_place.py`（需先按 README 安装 `galbot_sdk` 并在 G1 机器人环境 V1.18 上运行）。本仓库提供文档与 C++/Python 示例，`galbot_sdk` 库本体（GalbotRobot/GalbotMotion/GalbotNavigation 的实现）不在仓库内，下图只画示例脚本能确认的 API 调用顺序。

```mermaid
sequenceDiagram
    participant S as example6_pick_and_place.py main()
    participant R as galbot_sdk.g1.GalbotRobot
    participant M as galbot_sdk.g1.GalbotMotion
    participant N as galbot_sdk.g1.GalbotNavigation
    S->>S: check_robot_safety() 人工确认急停已释放
    S->>R: GalbotRobot()；init(enable_sensor_set)
    S->>M: GalbotMotion()；init()
    S->>N: GalbotNavigation()；init()
    S->>R: set_joint_positions 先腿部再头臂，回到初始位姿
    S->>N: is_localized()（未定位则跳过导航）
    S->>R: detect_object：get_rgb_data / get_depth_data，再 get_transform 把相机位姿转到 base_link
    S->>M: attach_tool(chain="left_arm", tool="galbot_gripper")
    S->>R: set_gripper_command 张开夹爪
    S->>M: set_end_effector_pose(pre_grasp_pose)
    S->>M: move_line(grasp_target) 直线逼近
    S->>R: set_gripper_command 闭合夹爪
    S->>M: set_end_effector_pose(lift) 后 set_end_effector_pose(retreat)
    S->>R: set_joint_positions 恢复左臂初始关节
    opt navigation_enabled
        S->>N: get_current_pose；check_path_reachability；navigate_to_goal(x+0.1m)；check_goal_arrival
    end
    S->>M: set_end_effector_pose(pre_place) 后 move_line(place_target)
    S->>R: set_gripper_command 张开放置
    S->>R: request_shutdown()；wait_for_shutdown()；destroy()
```

图依据：GalaxyGeneralRobotics/GalbotSDK@3691794，读过 README.md、examples/g1/python/tutorials/example6_pick_and_place.py（完整 main 与 pick_and_place 的调用部分）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [GalbotSDK 源码归档](../../sources/repos/galbotsdk.md)（<https://github.com/GalaxyGeneralRobotics/GalbotSDK>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：GalaxyGeneralRobotics/GalbotSDK@3691794，读过 README.md、examples/g1/python/tutorials/example6_pick_and_place.py；galbot_sdk 库本体不在仓库内；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [银河通用 官方组织](https://github.com/GalaxyGeneralRobotics)
