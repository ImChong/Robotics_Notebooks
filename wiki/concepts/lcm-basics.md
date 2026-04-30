---
type: concept
tags: [middleware, software, lcm, communication, realtime]
status: complete
updated: 2026-04-21
related:
  - ../comparisons/ros2-vs-lcm.md
  - ../queries/real-time-control-middleware-guide.md
  - ../formalizations/control-loop-latency-modeling.md
sources:
  - ../../sources/papers/sim2real.md
summary: "LCM（Lightweight Communications and Marshalling）是一款极致轻量的机器人中间件，以低延迟 UDP 组播为核心，是高性能足式机器人底层运控的事实标准。"
---

# LCM (Lightweight Communications and Marshalling) 基础

**LCM** 是一款由 MIT 团队开发的通信库，专门针对**高频、低延迟、高带宽**的机器人控制场景设计。在人形机器人和四足机器人的“脊髓级”控制中，LCM 是优于 ROS 2 的首选方案。

## 核心设计哲学

1. **UDP 组播 (Multicast)**：数据像无线电广播一样发出去，谁需要谁就收。没有 TCP 的握手和重传，保证了**最新数据**的绝对实时性。
2. ** मार्शल (Marshalling)**：提供了一套类似 JSON 但更紧凑的 IDL（接口描述语言），自动生成 C++/Python/Java 的强类型代码。
3. **零配置**：不需要后台进程（Daemon），只要在同一个局域网内，程序跑起来就能互相通信。

## 为什么高性能运控选 LCM？

在 [1kHz 的实时控制环路](../queries/real-time-control-middleware-guide.md)中：
- ROS 2/DDS 的复杂 QoS 检查会带来 2-5ms 的随机抖动（Jitter）。
- LCM 的延迟通常稳定在 **< 100微秒**。

## 关联页面
- [ROS 2 vs LCM 选型对比](../comparisons/ros2-vs-lcm.md)
- [实时运控中间件配置指南](../queries/real-time-control-middleware-guide.md)
- [控制环路延迟建模](../formalizations/control-loop-latency-modeling.md)

## 参考来源
- Huang, A. S., et al. (2010). *LCM: Lightweight Communications and Marshalling*.
