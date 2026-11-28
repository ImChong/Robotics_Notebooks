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
  - ../../sources/repos/roh_gen2_firmware.md
summary: "傲意科技 开源项目 roh_gen2_firmware（工程与工具）：ROHand Gen2 固件与文档…"
institutions:
  - ohand
---

# roh_gen2_firmware

## 一句话定义

**roh_gen2_firmware** 是 [傲意科技](https://github.com/oymotion) 公开的 **工程与工具** 开源项目：ROHand Gen2 固件与文档

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
| 类别 | 工程与工具 |
| 官方组织 | https://github.com/oymotion |

### 架构流程图

```mermaid
flowchart LR
  HW["ROH-AP001 / ROH-AP002 灵巧手硬件"] --> FW["firmware (板载固件)"]
  FW <--> PROTO["protocol (通信协议文档)"]
  PROTO <--> APP["OHandSetting (桌面端配置应用)"]
  PROTO <--> ROS["ROS / ROS2 驱动与 URDF (rohand_gen2_ros/ros2_pkg, urdf_ros1/ros2)"]
  ROS --> DEMO["应用示例 (roh_gen2_demos / roh_with_rm65 机械臂联调)"]
  DOCS["UserManual / FAQs (用户手册与常见问题)"] --> APP
```

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `roh_gen2_firmware` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 源码运行时序图

基于 `oymotion/roh_gen2_firmware` 仓库已核查的 `protocol/roh_registers_v2.py`、`protocol/roh_registers_v2.h`、`OHandSetting/Ubuntu20/OHandSetting.sh` 与 `FAQs/FAQs_CN.md`（附录 1、3、5、7），桌面配置脚本与上位机 Modbus-RTU 控制及力控读取流程如下（板载固件 C 源码未包含在本次核查文件中）。

```mermaid
sequenceDiagram
  autonumber
  participant Host as 上位机脚本 (FAQs_CN.md)
  participant Shell as OHandSetting.sh (Ubuntu20)
  participant Client as ModbusSerialClient (pymodbus RTU)
  participant Regs as roh_registers_v2 (.py / .h)
  participant Hand as ROHand 控制器 (NODE_ID=2)

  opt 桌面端 PID 调参入口
    Shell->>Shell: 配置 LD_LIBRARY_PATH 与 QT_PLUGIN_PATH
    Shell->>Hand: 启动 $BASE_DIR/bin/OHandSetting
  end

  Host->>Host: find_comport("CH340") 定位串口
  Host->>Client: ModbusSerialClient(..., FramerType.RTU, 115200).connect()
  Host->>Regs: 引用寄存器常量 (MODBUS_PROTOCOL_VERSION_MAJOR=2)

  Host->>Client: write_registers(ROH_FINGER_SPEED0, ..., NODE_ID)
  Client->>Hand: Modbus-RTU 写速度寄存器 (1125)
  Host->>Client: write_registers(ROH_FINGER_POS_TARGET0, ..., NODE_ID)
  Client->>Hand: 写目标位置寄存器 (1135..1140)
  Host->>Client: read_holding_registers(ROH_FINGER_STATUS0, 5, NODE_ID)
  Client->>Hand: 轮询堵转状态寄存器 (1085..1089)
  Hand-->>Host: 若 status==5 则读 ROH_FINGER_POS0+i (1145) 防过热

  opt 力控模式 (FAQs_CN.md 附录 5)
    Host->>Client: write_registers(ROH_RESET_FORCE, 1)
    Client->>Hand: 复位力传感器模块 (1016)
    Host->>Client: write_registers(ROH_FINGER_FORCE_TARGET0, [500,...])
    Client->>Hand: 写目标力寄存器 (1115)
    Host->>Client: read_holding_registers(ROH_FINGER_FORCE0 / ROH_FINGER_FORCE_EX0)
    Hand-->>Host: 返回合力 (1175) 与点阵单点力 (2000 + i*FORCE_GROUP_SIZE)
    Host->>Client: write_registers(ROH_FINGER_FORCE_TARGET0, [0,...])
  end

  Host->>Client: client.close()
```

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)
- [Robotic Manipulation Systems · 机器人操作系统与末端执行器](../methods/robotic-manipulation-systems.md)：对照灵巧手末端执行器的固件寄存器协议、位置/力控接口与 ROS 驱动系统集成。
- [Direct Force Feedback Control（直接力反馈控制）](../methods/direct-force-feedback-control.md)：参考上位机基于 Modbus-RTU 目标力与力传感器寄存器实现的直接力反馈控制。

## 参考来源

- [roh_gen2_firmware 源码归档](../../sources/repos/roh_gen2_firmware.md)（<https://github.com/oymotion/roh_gen2_firmware>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)

## 推荐继续阅读

- [傲意科技 官方组织](https://github.com/oymotion)
