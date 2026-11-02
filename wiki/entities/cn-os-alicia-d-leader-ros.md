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
  - ../../sources/repos/alicia-d-leader-ros.md
summary: "玄雅科技 开源项目 Alicia-D-Leader-ROS（遥操作与数据采集）：Alicia-D 主从遥操作（Leader）ROS 包…"
institutions:
  - xuanya
---

# Alicia-D-Leader-ROS

## 一句话定义

**Alicia-D-Leader-ROS** 是 [玄雅科技](https://github.com/Synria-Robotics) 公开的 **遥操作与数据采集** 开源项目：Alicia-D 主从遥操作（Leader）ROS 包

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
| 类别 | 遥操作与数据采集 |
| 官方组织 | https://github.com/Synria-Robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Alicia-D-Leader-ROS` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`roslaunch alicia_duo_leader_driver serial_server.launch`（同时拉起 `serial_server_node`、`serial_data_type_node.py`、`servo_states_node.py`、`servo_control_node.py`）；示例订阅端 `rosrun alicia_duo_leader_driver arm_read_demo.py`。

```mermaid
sequenceDiagram
    participant HW as 主端机械臂（串口）
    participant SS as serial_server_node（C++，readFrameThread）
    participant DT as serial_data_type_node.py
    participant SN as servo_states_node.py
    participant CT as servo_control_node.py
    participant DM as arm_read_demo.py
    SS->>HW: connectSerial() 打开串口（port 为空则 findSerialPort() 找 ttyUSB）
    HW-->>SS: 字节流（帧头 0xAA … 帧尾 0xFF）
    SS->>SS: 按 data[2]+5 判帧长，serialDataCheck() 校验和
    SS->>DT: 发布 /read_serial_data（UInt8MultiArray）
    alt 指令 ID data[1]==0x02（夹爪/按钮）
        DT->>DT: 取夹爪原始值，(raw-2048)/8.52 度转弧度
        DT->>SN: 发布 /gripper_angle（Float32MultiArray：弧度, but1, but2）
    else 指令 ID==0x04（舵机状态）
        DT->>SN: 转发 /servo_states
    else 指令 ID==0x06 / 0xEE
        DT-->>DT: 分别发布 /servo_states_6、/error_frame_deal（仓库内无订阅者）
    end
    SN->>SN: servo_states_callback：节流、核对舵机数、u8_array_to_rad、servo_to_joint_map 映射
    SN->>DM: 发布 /arm_joint_state（ArmJointState，弧度）
    SN-->>SN: 同时发布 /servo_states_main（Float32MultiArray）
    DM->>DM: arm_state_callback 打印关节/夹爪/按钮
    Note over CT,SS: 反向通路：/zero_calibrate(Bool) → zero_calib_callback → frame_ge(CMD_ZERO_CAL)
    CT->>SS: 发布 /send_serial_data
    SS->>HW: sendSerialDataCallback → serial_port_.write()
```

图依据：Synria-Robotics/Alicia-D-Leader-ROS@dd971f1，读过 README.md、alicia_duo_leader_driver/launch/serial_server.launch、src/serial_server_node_main.cpp、src/serial_server_node.cpp（部分）、scripts/serial_data_type_node.py、servo_states_node.py、servo_control_node.py（部分）、arm_read_demo.py（部分）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Alicia-D-Leader-ROS 源码归档](../../sources/repos/alicia-d-leader-ros.md)（<https://github.com/Synria-Robotics/Alicia-D-Leader-ROS>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Synria-Robotics/Alicia-D-Leader-ROS@dd971f1，读过 README.md、alicia_duo_leader_driver/launch/serial_server.launch、src/serial_server_node_main.cpp、src/serial_server_node.cpp、scripts/serial_data_type_node.py、scripts/servo_states_node.py、scripts/servo_control_node.py、scripts/arm_read_demo.py；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [玄雅科技 官方组织](https://github.com/Synria-Robotics)
