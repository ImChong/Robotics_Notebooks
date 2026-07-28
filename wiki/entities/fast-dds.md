---
type: entity
tags: [middleware, dds, ros2, rmw, eprosima, realtime, systems-engineering]
status: complete
updated: 2026-07-28
related:
  - ../concepts/dds-communication.md
  - ./cyclone-dds.md
  - ../concepts/ros2-basics.md
  - ../comparisons/ros2-vs-lcm.md
  - ../queries/real-time-control-middleware-guide.md
sources:
  - ../../sources/repos/fast-dds.md
  - ../../sources/sites/fast-dds-docs.md
  - ../../sources/sites/omg-dds-spec.md
  - ../../sources/repos/ros2.md
summary: "eProsima Fast DDS：OMG DDS/RTPS 的 Apache-2.0 C++ 实现；ROS 2 默认 RMW 之一（rmw_fastrtps）；双层 API、UDP/TCP/SHM 与 Discovery Server。"
---

# Fast DDS（eProsima）

## 一句话定义

**Fast DDS**（原 Fast RTPS）是 eProsima 的开源 **OMG DDS / RTPS** C++ 实现：提供 DCPS API 与底层 RTPS 访问，经 `rmw_fastrtps` 成为 ROS 2 最常用的默认中间件之一。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DDS | Data Distribution Service | OMG 数据分发服务标准 |
| RTPS | Real-Time Publish-Subscribe | DDS 互操作线协议 |
| RMW | ROS Middleware | 经 `rmw_fastrtps_cpp` 对接 |
| SHM | Shared Memory | 同机零拷贝类传输选项 |
| IDL | Interface Definition Language | Fast DDS-Gen 输入 |

## 为什么重要

- ROS 2 多数发行版默认走 Fast DDS；节点「连不上 / 抖动」首先查 **RMW + QoS + 发现**，不是业务逻辑。
- 相对 [Cyclone DDS](./cyclone-dds.md)：同属合规 RTPS，但默认发现策略、XML 配置面与 Pro 扩展不同——**仓库必须钉死一种**。
- 社区版与 **Fast DDS Pro** 能力不同：TSN、低带宽、IP Mobility 等属商业层，勿按开源文档假设可用。

## 核心原理

| 层级 | 内容 |
|------|------|
| DDS API | Domain / Topic / Writer / Reader + QoS |
| RTPS API | 更细的 Writer/Reader 与协议旋钮 |
| 传输 | UDPv4/v6、TCPv4/v6、SHM |
| 发现 | 默认动态发现；可配 Discovery Server |
| 代码生成 | Fast DDS-Gen：IDL → 类型与桩代码 |
| 序列化 | Fast CDR |

```mermaid
flowchart LR
  App[应用 / rclcpp] --> RMW[rmw_fastrtps]
  RMW --> FD[Fast DDS]
  FD --> RTPS[RTPS]
  RTPS --> T[UDP / TCP / SHM]
```

## 工程实践

1. ROS 2：`export RMW_IMPLEMENTATION=rmw_fastrtps_cpp`（或发行版默认即此）。
2. 将 **XML QoS / Discovery Server** 配置纳入版本库；多机无组播时优先配 peers 或 Discovery Server。
3. 高频状态：Best Effort + Keep Last(1)；命令通道 Reliable 且降频。
4. 独立使用（非 ROS）：按 [官方安装文档](https://fast-dds.docs.eprosima.com/) 装二进制或源码；可用 Fast DDS Suite Docker 做 HelloWorld / ShapesDemo。
5. 与 Cyclone **同域互通**前，先对齐 Topic 类型、QoS 兼容性与发现可达性。

**上游元数据（2026-07）：** [eProsima/Fast-DDS](https://github.com/eProsima/Fast-DDS) ~2.9k★，Apache-2.0，最新发行 v3.6.2；声称 ROS Quality Level 1。

## 局限与风险

- 动态分配与多线程模型仍可能制造尾延迟——**1 kHz 力矩环不要默认走 ROS 2 Topic**。
- QoS 不匹配会 **静默不连通**。
- Pro 功能不可用时勿按营销页选型。
- 开源状态：**已开源**（社区版）；Pro 为商业扩展。

## 关联页面

- [DDS 通信机制](../concepts/dds-communication.md)
- [Cyclone DDS](./cyclone-dds.md)
- [ROS 2 基础](../concepts/ros2-basics.md)
- [ROS 2 vs LCM](../comparisons/ros2-vs-lcm.md)
- [实时运控中间件指南](../queries/real-time-control-middleware-guide.md)

## 参考来源

- [sources/repos/fast-dds.md](../../sources/repos/fast-dds.md)
- [sources/sites/fast-dds-docs.md](../../sources/sites/fast-dds-docs.md)
- [sources/sites/omg-dds-spec.md](../../sources/sites/omg-dds-spec.md)
- [sources/repos/ros2.md](../../sources/repos/ros2.md)（`ros2.repos` 钉定）

## 推荐继续阅读

- 文档：<https://fast-dds.docs.eprosima.com/>
- 仓：<https://github.com/eProsima/Fast-DDS>
- OMG DDS 1.4：<https://www.omg.org/spec/DDS/1.4>
