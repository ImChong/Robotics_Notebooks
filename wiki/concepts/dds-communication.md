---
type: concept
tags: [systems-engineering, dds, ros2, middleware, qos, realtime, eprosima, eclipse, omg]
status: complete
updated: 2026-07-28
related:
  - ./ros2-basics.md
  - ./rmw-interface.md
  - ./remote-procedure-call.md
  - ../entities/grpc.md
  - ../entities/fast-dds.md
  - ../entities/cyclone-dds.md
  - ../comparisons/ros2-vs-lcm.md
  - ./lcm-basics.md
  - ./network-protocol-stack.md
  - ./message-queue-reliability.md
  - ../overview/topic-systems-engineering.md
  - ../overview/topic-communication.md
  - ../queries/real-time-control-middleware-guide.md
sources:
  - ../../sources/sites/omg-dds-spec.md
  - ../../sources/sites/fast-dds-docs.md
  - ../../sources/sites/cyclonedds-io.md
  - ../../sources/repos/fast-dds.md
  - ../../sources/repos/cyclonedds.md
  - ../../sources/repos/rmw.md
  - ../../sources/sites/ros2-design-rmw-interface.md
  - ../../sources/sites/ros2-rmw-middleware-vendors.md
  - ../../sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md
  - ../../sources/sites/ros2-official-documentation.md
  - ../../sources/repos/ros2.md
summary: "DDS（Data Distribution Service）：OMG DCPS/QoS 与 DDSI-RTPS 线协议；ROS 2 经 RMW 使用 Fast DDS / Cyclone DDS 等实现，解释可靠性与实时性权衡。"
---

# DDS 通信机制（Data Distribution Service）

## 一句话定义

**DDS** 是 OMG 标准化的 **数据中心化发布订阅** 中间件：用 Topic、类型系统与 **QoS** 在去中心化发现下分发数据；线上互操作靠 **DDSI-RTPS**。ROS 2 通过 RMW 使用其实现（[Fast DDS](../entities/fast-dds.md)、[Cyclone DDS](../entities/cyclone-dds.md) 等）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DDS | Data Distribution Service | 数据分发服务标准（OMG） |
| DCPS | Data-Centric Publish-Subscribe | 数据中心化发布订阅模型 |
| RTPS | Real-Time Publish-Subscribe | DDS 互操作线协议（DDSI-RTPS） |
| QoS | Quality of Service | 可靠性、历史、截止期等策略 |
| RMW | ROS Middleware | ROS 2 与具体 DDS 实现的适配层（专页：[rmw-interface](./rmw-interface.md)） |
| SPDP/SEDP | Simple Participant/Endpoint Discovery Protocol | RTPS 内置发现机制 |

## 为什么重要

- [ROS 2](./ros2-basics.md) 的延迟与丢包行为 **几乎都由底层 DDS QoS + 发现 + 传输** 决定。
- 把「ROS 2 Topic」当成魔法管道，调不好 1 kHz 环——对比见 [ROS 2 vs LCM](../comparisons/ros2-vs-lcm.md)。
- 标准层（OMG）与实现层（Fast / Cyclone）必须分开查：互通问题查 **RTPS + QoS 兼容**；性能问题查 **具体 vendor 配置与版本钉定**。

## 核心原理

### 标准两层（一手规范）

| 层 | 规范 | 回答什么 |
|----|------|----------|
| **API / 语义** | [OMG DDS 1.4](https://www.omg.org/spec/DDS/1.4)（2015） | Domain、实体、Topic、类型、QoS 契约 |
| **线协议** | [DDSI-RTPS 2.5](https://www.omg.org/spec/DDSI-RTPS/2.5)（2022） | 报文、发现（SPDP/SEDP）、与 Writer/Reader 映射 |

归档：[omg-dds-spec](../../sources/sites/omg-dds-spec.md)。

### 实体与匹配

1. **实体**：DomainParticipant、Publisher/Subscriber、DataWriter/DataReader、Topic。
2. **发现**：默认 UDP 组播发现对端（环境无组播时需配置 peers / Discovery Server）。
3. **QoS 关键项**：
   - Reliability：Best Effort vs Reliable
   - History：Keep Last(N) vs Keep All
   - Durability：Volatile vs Transient Local…
   - Deadline / Liveliness：周期与存活监测
4. **传输**：RTPS 常跑在 UDP 上；Reliable 会引入重传与抖动；实现还可提供 TCP / SHM。

```mermaid
flowchart LR
  NodeA[ROS 2 Node A] --> RMW[RMW]
  RMW --> DDS[DDS Vendor]
  DDS --> RTPS[RTPS over UDP]
  RTPS --> DDS2[DDS Vendor]
  DDS2 --> RMW2[RMW]
  RMW2 --> NodeB[ROS 2 Node B]
```

### ROS 2 常用实现（一手仓）

| 实现 | 组织 | ROS 2 RMW | 许可 | 备注 |
|------|------|-----------|------|------|
| [Fast DDS](../entities/fast-dds.md) | eProsima | `rmw_fastrtps_cpp` | Apache-2.0 | 多数 LTS 默认之一；双层 DDS+RTPS API |
| [Cyclone DDS](../entities/cyclone-dds.md) | Eclipse | `rmw_cyclonedds_cpp` | EPL-2.0 / EDL-1.0 | tier-1；Unitree 真机常用 |
| （实验）Zenoh 等 | — | `rmw_zenoh` 等 | — | 见 `ros2.repos`；非本页展开 |

二者均由 [ros2/ros2](../../sources/repos/ros2.md) 的 `ros2.repos` 钉定版本。

## 工程实践

- **高频状态**：Best Effort + Keep Last(1)，只要最新。
- **关键命令**：Reliable，但降低频率，或改走服务/共享内存/[LCM](./lcm-basics.md)。
- 固定 RMW vendor 与 XML/QoS 配置进仓库；记录 `RMW_IMPLEMENTATION`。
- 大规模机器人：限制发现流量、分区 Domain、避免不必要的大消息。
- 多实现互通：先对齐 **Domain ID、类型、QoS 兼容、发现可达**；再谈性能。
- Unitree 栈：优先 Cyclone，并钉发行线（见 [cyclone-dds](../entities/cyclone-dds.md)、[unitree_ros2](../entities/unitree-ros2.md)）。

## 局限与风险

- QoS 不兼容的 Writer/Reader **静默不连通**。
- 动态分配与多线程在部分实现中影响尾延迟——**不要把 1 kHz 关节环默认丢在 DDS topic 上**（[运控指南](../queries/real-time-control-middleware-guide.md)）。
- 与 [消息队列](./message-queue-reliability.md) 不同：默认不提供长期积压与跨周审计日志。
- 「开了 ROS 2」≠「理解了 DDS」：发行版默认 vendor 会变，升级 distro 需回归通信。

## 关联页面

- [ROS 2 基础](./ros2-basics.md)
- [RMW 接口](./rmw-interface.md)
- [远程过程调用（RPC）](./remote-procedure-call.md)（请求/响应 vs 本页 pub/sub）
- [gRPC](../entities/grpc.md)
- [Fast DDS](../entities/fast-dds.md)
- [Cyclone DDS](../entities/cyclone-dds.md)
- [ROS 2 vs LCM](../comparisons/ros2-vs-lcm.md)
- [LCM 基础](./lcm-basics.md)
- [网络协议栈](./network-protocol-stack.md)
- [通信协议专题](../overview/topic-communication.md)
- [实时运控中间件指南](../queries/real-time-control-middleware-guide.md)

## 参考来源

- [OMG DDS / DDSI-RTPS 规范](../../sources/sites/omg-dds-spec.md)
- [Fast DDS 文档](../../sources/sites/fast-dds-docs.md) · [仓](../../sources/repos/fast-dds.md)
- [Cyclone DDS 官网](../../sources/sites/cyclonedds-io.md) · [仓](../../sources/repos/cyclonedds.md)
- [DDS/RTOS/边云合集索引](../../sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md)
- [ROS 2 官方文档](../../sources/sites/ros2-official-documentation.md) · [ros2 元仓](../../sources/repos/ros2.md)
- [ros2/rmw](../../sources/repos/rmw.md) · [RMW Design](../../sources/sites/ros2-design-rmw-interface.md) · [Vendors / 多 RMW](../../sources/sites/ros2-rmw-middleware-vendors.md)

## 推荐继续阅读

- OMG DDS 1.4：<https://www.omg.org/spec/DDS/1.4>
- OMG DDSI-RTPS 2.5：<https://www.omg.org/spec/DDSI-RTPS/2.5>
- Fast DDS 文档：<https://fast-dds.docs.eprosima.com/>
- Cyclone DDS：<https://cyclonedds.io/>
