---
type: concept
tags: [systems-engineering, networking, tcp, udp, http, tls, dns, load-balancing]
status: complete
updated: 2026-07-28
related:
  - ./operating-system-basics.md
  - ./dds-communication.md
  - ./remote-procedure-call.md
  - ../entities/grpc.md
  - ../formalizations/udp-multicast-dynamics.md
  - ./lcm-basics.md
  - ../overview/hub-systems-engineering.md
sources:
  - ../../sources/sites/systems_engineering_os_network_primary_refs.md
  - ../../sources/sites/grpc-io-docs.md
  - ../../sources/sites/rfc-5531-onc-rpc.md
summary: "网络协议栈基础（TCP、UDP、HTTP、DNS、TLS、负载均衡）：区分机器人可靠回传面与实时数据面，避免把 TCP/TLS 语义误用于力矩环。"
---

# 网络协议栈基础（TCP / UDP / HTTP / DNS / TLS / 负载均衡）

## 一句话定义

**网络协议栈基础** 给出机器人研发与部署中最常用的传输/应用层协议角色划分：**哪些适合可靠服务，哪些适合低延迟传感/控制数据**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TCP | Transmission Control Protocol | 面向连接、可靠字节流 |
| UDP | User Datagram Protocol | 无连接数据报，可丢包 |
| HTTP | Hypertext Transfer Protocol | 应用层请求/响应语义 |
| DNS | Domain Name System | 名称到地址解析 |
| TLS | Transport Layer Security | 传输层加密与身份 |
| LB | Load Balancer | 流量分发到多后端 |

## 为什么重要

- 遥操作视频、模型下载、OTA、实验 API 走 **TCP + TLS + HTTP**。
- LCM、多数 DDS RTPS、自定义关节状态流走 **UDP**（常组播）——见 [UDP 组播动力学](../formalizations/udp-multicast-dynamics.md)。
- 误用 TCP 重传会把「丢一帧」变成「迟到一串」，在控制环上比丢包更危险。

## 核心原理

| 协议 | 保证 | 典型机器人用途 | 忌用场景 |
|------|------|----------------|----------|
| **UDP** | 尽力而为 | LCM、DDS、传感广播 | 需要强顺序持久化的账单/审计 |
| **TCP** | 可靠、有序 | SSH、日志汇聚、权重下载 | 1 kHz 力矩、最新状态话题 |
| **HTTP(S)** | 请求语义 +（常）TLS | 机器人服务 API、模型仓库 | 硬实时 |
| **HTTP/2 + gRPC** | 多路复用 RPC | 边云推理 / 工具 API（见 [gRPC](../entities/grpc.md)） | 1 kHz 控制环 |
| **DNS** | 名称解析 | 服务发现前置 | 完全离线现场需静态配置 |
| **TLS** | 机密性与完整性 | 云边通道、OTA | 握手成本不可进控制环 |
| **LB** | 分发与健康检查 | 训练集群、仿真 farm | 机载实时总线 |

## 工程实践

1. **分层**：机载实时网（UDP/总线）与办公/云网（TCP/HTTP）物理或 VLAN 隔离。
2. **QoS 意识**：DDS Reliability 若设 RELIABLE，底层仍可能用 TCP-like 行为——见 [DDS](./dds-communication.md)。
3. **超时**：所有云调用设硬超时；失败走本地退化策略，而非无限重试堵死线程。
4. **LB**：仅放在非实时服务前；健康检查不要误杀正在做长训练的 Job。

## 局限与风险

- 「有线就等于低延迟」——交换机缓冲、巨型帧、错误的 NIC 中断合并仍会抖。
- DNS TTL 与证书过期是边缘断网后的隐性故障。
- 把 HTTP 长轮询当控制通道会引入队头阻塞。

## 关联页面

- [操作系统基础](./operating-system-basics.md)
- [LCM 基础](./lcm-basics.md)
- [DDS 通信机制](./dds-communication.md)
- [远程过程调用（RPC）](./remote-procedure-call.md)
- [gRPC](../entities/grpc.md)
- [系统工程知识链](../overview/hub-systems-engineering.md)

## 参考来源

- [OS 与网络一手资料](../../sources/sites/systems_engineering_os_network_primary_refs.md)
- [gRPC 官方文档](../../sources/sites/grpc-io-docs.md)
- [RFC 5531 ONC RPC](../../sources/sites/rfc-5531-onc-rpc.md)

## 推荐继续阅读

- IETF RFC 768 / 9293 / 9110 / 8446
- RFC 5531：<https://www.rfc-editor.org/rfc/rfc5531>
