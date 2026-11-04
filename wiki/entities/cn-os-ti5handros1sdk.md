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
  - ../../sources/repos/ti5handros1sdk.md
summary: "钛虎机器人 开源项目 Ti5HandROS1SDK（SDK/驱动）：钛虎五指灵巧手 ROS1 SDK…"
institutions:
  - ti5robot
---

# Ti5HandROS1SDK

## 一句话定义

**Ti5HandROS1SDK** 是 [钛虎机器人](https://github.com/ti5robot) 公开的 **SDK/驱动** 开源项目：钛虎五指灵巧手 ROS1 SDK

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

1. 从官方 GitHub/Gitee 组织检索 `Ti5HandROS1SDK` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

入口：终端1 `rosrun hand_control hand_control_node`（串口控制节点）；终端2 `rosrun hand_control keyboard_publisher`（键盘发布节点）

```mermaid
sequenceDiagram
    autonumber
    participant U as 操作者（终端2 键盘输入）
    participant KP as keyboard_publisher 节点
    participant T as 话题 hand_command（Int32MultiArray）
    participant HN as hand_control_node 节点
    participant RS as RS485Controller（/dev/ttyUSB0，115200）
    participant D as 灵巧手硬件（仓库外）

    U->>KP: 输入 7 个整数后回车
    KP->>KP: istringstream 解析，凑满 7 个才 publish
    KP->>T: publish(msg.data = 7 个整数)
    T->>HN: commandCallback(msg)，取 data[0] 作为 command
    alt command 为 1 或 2（预设握手 / 松手）
        HN->>RS: set_servo_position(command, msg, 950)
        RS->>D: send_command 写入 18 字节帧（地址 0x03，模式 0xa3）
        Note over RS: usleep 2.5 秒后发送第二帧，松手分支第二帧全 0
    else command 为 3（按 data[1..6] 逐指位置控制）
        HN->>RS: set_servo_position(3, msg, 950)，速度覆盖为 1000
        loop i = 1 到 6
            RS->>RS: build_command(0x03, 0xA1, {i, 0x01, 位置, 速度低字节, 速度高字节, 0})
            RS->>D: write 18 字节帧，失败则返回 false
            Note over RS: usleep 5 毫秒
        end
    else command 为 4 到 6 或其它
        Note over HN: 无对应分支，直接返回
    end
    Note over RS,D: 读回路径被注释（read_bytes 恒为 0），仅校验写入是否完整，不验证硬件响应
```

图依据：ti5robot/Ti5HandROS1SDK@d56aaeb，hand_ws/src/hand_control/src/hand_control_node.cpp、hand_ws/src/hand_control/src/keyboard_publisher.cpp、hand_ws/src/hand_control/CMakeLists.txt、README.md

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Ti5HandROS1SDK 源码归档](../../sources/repos/ti5handros1sdk.md)（<https://github.com/ti5robot/Ti5HandROS1SDK>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：ti5robot/Ti5HandROS1SDK@d56aaeb，读过 README.md、hand_ws/src/hand_control/CMakeLists.txt、hand_ws/src/hand_control/src/hand_control_node.cpp、hand_ws/src/hand_control/src/keyboard_publisher.cpp；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [钛虎机器人 官方组织](https://github.com/ti5robot)
