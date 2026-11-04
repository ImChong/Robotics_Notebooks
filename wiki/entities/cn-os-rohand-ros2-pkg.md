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
  - ../../sources/repos/rohand_ros2_pkg.md
summary: "傲意科技 开源项目 rohand_ros2_pkg（SDK/驱动）：ROHand 仿生灵巧手 ROS2 包…"
institutions:
  - ohand
---

# rohand_ros2_pkg

## 一句话定义

**rohand_ros2_pkg** 是 [傲意科技](https://github.com/oymotion) 公开的 **SDK/驱动** 开源项目：ROHand 仿生灵巧手 ROS2 包

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
| 机构 | 傲意科技 |
| 类别 | SDK/驱动 |
| 官方组织 | https://github.com/oymotion |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `rohand_ros2_pkg` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：`ros2 run rohand rohand_modbus_a001 --ros-args -p port_name:="/dev/ttyUSB0" -p baudrate:=115200 -p hand_ids:=[2]`（README §5.2；`setup.py` console_scripts 指向 `scripts.roh_a001_node.rohand_modbus:main`）

```mermaid
sequenceDiagram
    participant Ros as ROS2 上层（发布 target_joint_states）
    participant Main as rohand_modbus.py main()
    participant Node as ROHandNode(Node)
    participant Mb as pymodbus ModbusSerialClient
    participant Hand as ROHand 从站（hand_ids）
    Main->>Main: rclpy.init()
    Main->>Node: ROHandNode()
    Node->>Node: declare_parameters: port_name / baudrate / hand_ids
    Node->>Node: create_subscription(~/target_joint_states)，create_publisher(~/current_joint_states)，create_rate(30)
    Node->>Mb: ModbusSerialClient(port, baudrate).connect()
    loop 最多 10 次
        Node->>Mb: read_holding_registers(ROH_PROTOCOL_VERSION)
        Mb->>Hand: Modbus-RTU 读请求
        Hand-->>Mb: 协议版本寄存器
        Mb-->>Node: rr
        Note over Node: 主版本与 MODBUS_PROTOCOL_VERSION_MAJOR 不符则 raise
    end
    Node->>Node: 启动 _thread_pub 线程
    Main->>Node: rclpy.spin(node)
    Ros->>Node: JointState（_joint_states_callback）
    Node->>Mb: write_registers(ROH_FINGER_ANGLE_TARGET0, 角度×100)
    Mb->>Hand: 写目标角度
    loop _thread_pub，while rclpy.ok()，每个 hand_id
        Node->>Mb: read_holding_registers(ROH_FINGER_ANGLE0, 6)
        Mb->>Hand: 读当前角度
        Hand-->>Node: 6 个关节角度（÷100 为 position）
        Node->>Mb: read_holding_registers(ROH_FINGER_FORCE0, 6)
        Mb->>Hand: 读当前力量
        Hand-->>Node: 6 个力量值（写入 effort）
        Node-->>Ros: publish ~/current_joint_states
    end
```

图依据：rohand_ros2_pkg@12f7054，`README.md`、`setup.py`、`scripts/roh_a001_node/rohand_modbus.py`、`common/roh_registers_v1.py`；未运行代码。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [rohand_ros2_pkg 源码归档](../../sources/repos/rohand_ros2_pkg.md)（<https://github.com/oymotion/rohand_ros2_pkg>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：oymotion/rohand_ros2_pkg@12f7054，读过 README.md、setup.py、scripts/roh_a001_node/rohand_modbus.py、common/roh_registers_v1.py（另抽读 scripts/roh_ap001_node/rohand_modbus.py 的前半部分）；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [傲意科技 官方组织](https://github.com/oymotion)
