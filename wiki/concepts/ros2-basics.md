---
type: concept
tags: [middleware, software, ros2, architecture, deployment]
status: complete
updated: 2026-04-21
related:
  - ../comparisons/ros2-vs-lcm.md
  - ../queries/real-time-control-middleware-guide.md
sources:
  - ../../sources/papers/sim2real.md
summary: "ROS 2（Robot Operating System 2）是机器人开发的事实标准中间件，基于 DDS 协议实现了高可靠、模块化的分布式通信与丰富的算法生态支持。"
---

# ROS 2 (Robot Operating System 2) 基础

**ROS 2** 是全球机器人社区中最广泛使用的开源框架。它并非真正的操作系统，而是一套运行在 Linux 之上的**中间件 (Middleware)**，提供了标准化的通信协议、开发工具和海量的算法库。

## 核心架构：DDS 驱动

相比于 ROS 1，ROS 2 最核心的进化是采用了 **DDS (Data Distribution Service)** 作为底层通信。
- **去中心化**：不再依赖一个中心化的 Master 节点，单点故障不会导致全系统崩溃。
- **服务质量 (QoS)**：允许开发者针对每个 Topic 设置实时性要求（如“丢弃旧数据”或“必须送达”）。

## 核心组件

1. **节点 (Nodes)**：执行特定任务的独立进程。
2. **话题 (Topics)**：基于发布/订阅（Pub/Sub）模式的异步数据流。
3. **服务 (Services)**：基于请求/响应（Request/Response）模式的同步通信。
4. **动作 (Actions)**：用于长时运行的任务（如“移动到 B 点”），支持进度反馈和中途取消。

## 在运控中的地位

尽管 ROS 2 在处理 [1000Hz 底层环路](../comparisons/ros2-vs-lcm.md) 时可能存在延迟抖动，但它是**高层感知、路径规划和多传感器标定**的不二之选。

## 关联页面
- [ROS 2 vs LCM 选型对比](../comparisons/ros2-vs-lcm.md)
- [实时运控中间件配置指南](../queries/real-time-control-middleware-guide.md)

## 参考来源
- ROS 2 官方文档 (docs.ros.org).
