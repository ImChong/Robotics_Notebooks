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
  - ../../sources/repos/bessica-d-sdk.md
summary: "玄雅科技 开源项目 Bessica-D-SDK（SDK/驱动）：Bessica-D 机械臂 SDK…"
institutions:
  - xuanya
---

# Bessica-D-SDK

## 一句话定义

**Bessica-D-SDK** 是 [玄雅科技](https://github.com/Synria-Robotics) 公开的 **SDK/驱动** 开源项目：Bessica-D 机械臂 SDK

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

1. 从官方 GitHub/Gitee 组织检索 `Bessica-D-SDK` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python examples/05_demo_move_joint.py --port <串口>`（Python SDK，双臂关节运动示例；其余示例同样从 `bessica_d_sdk.create_robot()` 起步）。

```mermaid
sequenceDiagram
    participant EX as examples/05_demo_move_joint.py
    participant CR as bessica_d_sdk.create_robot（__init__.py）
    participant API as SynriaBessicaRobotAPI（api/synria_b_robot_api.py）
    participant SD as ServoDriver（hardware/servo_driver.py）
    participant SC as SerialComm（hardware/serial_comm.py）
    participant DP as DataParser（hardware/data_parser.py）
    participant HW as Bessica-D 双臂（串口）
    EX->>CR: create_robot(port, baudrate, robot_version, speed_deg_s)
    CR->>SD: ServoDriver(port, baudrate)
    CR->>API: SynriaBessicaRobotAPI(servo_driver, robot_version)
    API->>API: synriard.get_model_path("Bessica_D") → robocore RobotModel
    EX->>API: connect()
    API->>SD: connect()
    SD->>SC: connect()
    SD->>SD: start_update_thread()，wait_for_valid_state(arm="both")
    loop 后台线程 _update_loop（poll_mode）
        SD->>SC: send_data([0xAA,0x06,0x01,0x00,0x00,0xFF]) 关节状态请求
        HW-->>SC: 状态帧
        SD->>SC: read_frame()
        SD->>DP: parse_frame(frame)
    end
    EX->>API: set_speed(speed_deg_s)，set_home()
    EX->>API: set_joint_target(target_joints, arm="left_arm"/"right_arm")
    API->>SD: move_joints_deg(arm, angles_deg, wait_for_completion)
    SD->>SD: read_joint_state(arm) → set_joint_angles(...)
    SD->>SD: _build_joint_frame(mapped_angles, arm)
    SD->>SC: send_data(frame)
    SC->>HW: 串口写入关节目标
    EX->>API: set_gripper_target(arm="both", command="close")
    EX->>API: disconnect()
```

图依据：Synria-Robotics/Bessica-D-SDK@bc908e1，读过 README.md、bessica_d_sdk/__init__.py、api/synria_b_robot_api.py（部分）、hardware/servo_driver.py（部分）、examples/05_demo_move_joint.py；serial_comm.py、data_parser.py 未读，set_gripper_target 内部调用未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Bessica-D-SDK 源码归档](../../sources/repos/bessica-d-sdk.md)（<https://github.com/Synria-Robotics/Bessica-D-SDK>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Synria-Robotics/Bessica-D-SDK@bc908e1，读过 README.md、bessica_d_sdk/__init__.py、bessica_d_sdk/api/synria_b_robot_api.py（部分）、bessica_d_sdk/hardware/servo_driver.py（部分）、examples/05_demo_move_joint.py；serial_comm.py、data_parser.py 未读；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [玄雅科技 官方组织](https://github.com/Synria-Robotics)
