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
  - ../../sources/repos/alicia-d-sdk.md
summary: "玄雅科技 开源项目 Alicia-D-SDK（SDK/驱动）：Alicia-D 六轴机械臂（带夹爪）Python 控制 SDK（串口、关节/夹爪/轨迹控制与状态读取）…"
institutions:
  - xuanya
---

# Alicia-D-SDK

## 一句话定义

**Alicia-D-SDK** 是 [玄雅科技](https://github.com/Synria-Robotics) 公开的 **SDK/驱动** 开源项目：Alicia-D 六轴机械臂（带夹爪）Python 控制 SDK（串口、关节/夹爪/轨迹控制与状态读取）

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

1. 从官方 GitHub/Gitee 组织检索 `Alicia-D-SDK` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`python examples/05_demo_move_joint.py --port <串口>`（Python SDK；其余示例同样从 `alicia_d_sdk.create_robot()` 起步）。

```mermaid
sequenceDiagram
    participant EX as examples/05_demo_move_joint.py
    participant PK as alicia_d_sdk.create_robot（__init__.py）
    participant API as SynriaRobotAPI（api/synria_robot_api.py）
    participant SD as ServoDriver（hardware/servo_driver.py）
    participant SC as SerialComm（hardware/serial_comm.py）
    participant DP as DataParser（hardware/data_parser.py）
    participant HW as Alicia-D 机械臂（串口）
    EX->>PK: create_robot(port=args.port)
    PK->>SD: ServoDriver(port, debug_mode)，内部创建 SerialComm + DataParser
    PK->>PK: get_model_path("Alicia_D", version, variant) → robocore RobotModel
    PK->>API: SynriaRobotAPI(servo_driver, robot_model)，auto_connect=True
    API->>SD: connect()
    SD->>SC: connect()（port 为空则 find_serial_port）
    SD->>SD: start_update_thread()（_update_loop 后台线程）
    loop 后台线程 _update_loop
        SD->>SC: read_frame()
        SC-->>SD: 校验通过的帧
        SD->>DP: parse_frame(frame)
    end
    API->>SD: acquire_info("joint_gripper", wait=True) 并读取状态、_robot_type()
    HW-->>DP: 关节/夹爪数据
    EX->>API: set_home(speed_deg_s) → set_robot_state(target_joints, joint_format='deg', wait_for_completion=True)
    API->>SD: set_joint_and_gripper(joint_angles, gripper_value, speed_deg_s)
    SD->>SD: _build_joint_frame()
    SD->>SC: send_data(frame)
    SC->>HW: 串口写入目标位置与速度
    API->>API: _wait_for_joint_target()（轮询状态至到位或超时）
    EX->>API: disconnect()
```

图依据：Synria-Robotics/Alicia-D-SDK@7c49e07，读过 pyproject.toml、alicia_d_sdk/__init__.py、alicia_d_sdk/api/synria_robot_api.py、alicia_d_sdk/hardware/servo_driver.py、examples/05_demo_move_joint.py；serial_comm.py、data_parser.py 仅读到类与函数签名。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Alicia-D-SDK 源码归档](../../sources/repos/alicia-d-sdk.md)（<https://github.com/Synria-Robotics/Alicia-D-SDK>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Synria-Robotics/Alicia-D-SDK@7c49e07，读过 pyproject.toml、alicia_d_sdk/__init__.py、alicia_d_sdk/api/synria_robot_api.py、alicia_d_sdk/hardware/servo_driver.py、examples/05_demo_move_joint.py（部分），serial_comm.py 与 data_parser.py 仅读函数签名；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [玄雅科技 官方组织](https://github.com/Synria-Robotics)
