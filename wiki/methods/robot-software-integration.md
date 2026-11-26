---
type: method
tags: [robotics, software, integration]
status: complete
updated: 2026-11-26
summary: "将传感、通信、控制和应用模块连接为可运行、可诊断的机器人软件系统。"
sources:
  - ../../sources/repos/awesome-agentic-robots-survey-index.md
  - ../../sources/sites/platform-method-robot-software-integration.md
original:
  - ../../sources/sites/platform-method-robot-software-integration.md
related:
  - ../concepts/ros2-basics.md
  - robot-control-architecture.md
  - robot-sensor-integration.md
---

# Robot Software System Integration · 机器人软件系统集成

## 一句话定义

将传感、通信、控制和应用模块连接为可运行、可诊断的机器人软件系统。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| ROS | Robot Operating System | 机器人组件与通信生态 |
| DDS | Data Distribution Service | 常见 ROS 2 数据分发中间件规范 |
| QoS | Quality of Service | 通信可靠性、历史及持久性等策略 |
| HIL | Hardware-in-the-Loop | 真实设备参与闭环联调 |

## 为什么重要

能看到 ROS 图不代表订阅兼容，也不能保证程序在正确坐标或时间域消费数据。先写清接口和故障契约，再拼接节点。

## 主要技术路线（核心机制）

ROS 2 QoS 设计使用请求/提供模型匹配通信端点。历史深度、可靠性和持久性分别影响缓存、重传及晚加入订阅；偏好新鲜样本的感知流通常不能照搬另一条控制流的配置。

```mermaid
flowchart LR
    N0["定义消息时间与坐标契约"]
    N1["接入驱动和中间件"]
    N2["核对 QoS 与启动依赖"]
    N3["离线和仿真回归"]
    N4["硬件联调与故障验收"]
    N0 --> N1 --> N2 --> N3 --> N4
```

## 工程实践

建立话题/服务/动作接口表，记录消息、时间基准、单位、坐标和超期策略。离线回放检查对应时刻的 TF 与 QoS 匹配，随后在仿真测试启动次序、节点重启。接真机后独立测试断连、反馈过期、慢消费者和安全降级。

## 局限与风险

可靠消息不承诺硬实时，也不应让过期命令重新有效。ROS 2 是示例；换用其他中间件仍需要同样的时间和失效契约。

## 关联页面

- [ROS 2 基础](../concepts/ros2-basics.md)
- [机器人控制架构](robot-control-architecture.md)
- [机器人传感器集成](robot-sensor-integration.md)

## 参考来源

- [ROS 2 Design · Quality of Service Policies](../../sources/sites/platform-method-robot-software-integration.md)

## 推荐继续阅读

- [ROS 2 QoS 官方设计说明](https://design.ros2.org/articles/qos)
