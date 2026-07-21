---
type: concept
tags: [systems-engineering, dds, ros2, middleware, qos, realtime]
status: complete
updated: 2026-07-21
related:
  - ./ros2-basics.md
  - ../comparisons/ros2-vs-lcm.md
  - ./lcm-basics.md
  - ./network-protocol-stack.md
  - ./message-queue-reliability.md
  - ../overview/topic-systems-engineering.md
  - ../overview/topic-communication.md
sources:
  - ../../sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md
  - ../../sources/sites/ros2-official-documentation.md
summary: "DDS（Data Distribution Service）通信机制：OMG DCPS/QoS 与 RTPS 线协议；ROS 2 RMW 的底层，解释可靠性与实时性权衡。"
---

# DDS 通信机制（Data Distribution Service）

## 一句话定义

**DDS** 是 OMG 标准化的 **数据中心化发布订阅** 中间件：用 Topic、类型系统与 **QoS** 在去中心化发现下分发数据；ROS 2 通过 RMW 使用其实现（Fast DDS、Cyclone DDS 等）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DDS | Data Distribution Service | 数据分发服务标准 |
| DCPS | Data-Centric Publish-Subscribe | 数据中心化发布订阅模型 |
| RTPS | Real-Time Publish-Subscribe | DDS 互操作线协议 |
| QoS | Quality of Service | 可靠性、历史、截止期等策略 |
| RMW | ROS Middleware | ROS 2 与具体 DDS 实现的适配层 |

## 为什么重要

- [ROS 2](./ros2-basics.md) 的延迟与丢包行为 **几乎都由底层 DDS QoS + 发现 + 传输** 决定。
- 把「ROS 2 Topic」当成魔法管道，调不好 1 kHz 环——对比见 [ROS 2 vs LCM](../comparisons/ros2-vs-lcm.md)。

## 核心原理

1. **实体**：DomainParticipant、Publisher/Subscriber、DataWriter/DataReader、Topic。
2. **发现**：默认 UDP 组播发现对端（环境无组播时需配置 peers）。
3. **QoS 关键项**：
   - Reliability：Best Effort vs Reliable
   - History：Keep Last(N) vs Keep All
   - Durability：Volatile vs Transient Local…
   - Deadline / Liveliness：周期与存活监测
4. **传输**：RTPS 常跑在 UDP 上；Reliable 会引入重传与抖动。

```mermaid
flowchart LR
  NodeA[ROS 2 Node A] --> RMW[RMW]
  RMW --> DDS[DDS Vendor]
  DDS --> RTPS[RTPS over UDP]
  RTPS --> DDS2[DDS Vendor]
  DDS2 --> RMW2[RMW]
  RMW2 --> NodeB[ROS 2 Node B]
```

## 工程实践

- **高频状态**：Best Effort + Keep Last(1)，只要最新。
- **关键命令**：Reliable，但降低频率，或改走服务/共享内存。
- 固定 RMW vendor 与 XML QoS 配置进仓库；记录 `RMW_IMPLEMENTATION`。
- 大规模机器人：限制发现流量、分区 Domain、避免不必要的大消息。

## 局限与风险

- QoS 不兼容的 Writer/Reader **静默不连通**。
- 动态分配与多线程在部分实现中影响尾延迟。
- 与 [消息队列](./message-queue-reliability.md) 不同：默认不提供长期积压与跨周审计日志。

## 关联页面

- [ROS 2 基础](./ros2-basics.md)
- [ROS 2 vs LCM](../comparisons/ros2-vs-lcm.md)
- [网络协议栈](./network-protocol-stack.md)
- [通信协议专题](../overview/topic-communication.md)

## 参考来源

- [DDS/RTOS/边云/OTA/安全 FSM 一手资料](../../sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md)
- [ROS 2 官方文档归档](../../sources/sites/ros2-official-documentation.md)

## 推荐继续阅读

- OMG DDS 1.4：<https://www.omg.org/spec/DDS/1.4>
