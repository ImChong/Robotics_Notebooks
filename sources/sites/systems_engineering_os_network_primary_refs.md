# 操作系统与网络协议栈一手资料索引

> 来源归档（ingest）

- **标题：** OS 进程/调度/内存与 TCP/UDP/HTTP/DNS/TLS/负载均衡一手依据
- **类型：** book / RFC / standard（合集）
- **入库日期：** 2026-07-21
- **一句话说明：** 为人形/移动机器人「主控 Linux + 感知板 + 云端服务」提供操作系统与网络基线概念的原始依据，沉淀到系统工程专题。
- **沉淀到 wiki：** 是 → [operating-system-basics](../../wiki/concepts/operating-system-basics.md)、[network-protocol-stack](../../wiki/concepts/network-protocol-stack.md)、[topic-systems-engineering](../../wiki/overview/topic-systems-engineering.md)

## 为什么值得保留

- 真机运控抖动常被误诊为「策略问题」，根因往往是 **调度抢占、页面缺页、文件系统同步、TCP 重传**。
- 云边协同、遥操作、OTA、日志回传都依赖 **TCP/TLS/HTTP/DNS**；关节环路则应避开它们，改用 CAN/EtherCAT/UDP 组播。

## 核心摘录

### 1) OSTEP — Operating Systems: Three Easy Pieces

- **来源：** Remzi H. Arpaci-Dusseau & Andrea C. Arpaci-Dusseau, *[Operating Systems: Three Easy Pieces](https://ostep.org/)*（免费教材）。
- **要点：**
  - **进程 vs 线程**：进程是资源隔离边界（地址空间、文件描述符）；线程共享地址空间，适合同一控制进程内的传感/推理/日志分工。
  - **调度**：时间片、优先级、实时策略（FIFO/RR）决定延迟尾部；机器人 1 kHz 环应固定到实时线程并隔离 CPU。
  - **虚拟内存**：缺页、换页、TLB 未命中会引入毫秒级长尾——实时路径应避免冷路径分配与大块 map。
  - **文件系统**：日志落盘、`fsync`、NFS 抖动会饿死控制线程；运控与日志 I/O 必须分核/分队列。
- **对 wiki 的映射：** [operating-system-basics](../../wiki/concepts/operating-system-basics.md)、[rtos-realtime-scheduling](../../wiki/concepts/rtos-realtime-scheduling.md)

### 2) Linux 内核文档 — 调度与内存

- **来源：** [Linux Kernel Documentation — Scheduler](https://www.kernel.org/doc/html/latest/scheduler/)、[Memory Management](https://www.kernel.org/doc/html/latest/admin-guide/mm/index.html)；PREEMPT_RT 见 [Real-time Linux Wiki](https://wiki.linuxfoundation.org/realtime/start)。
- **要点：**
  - CFS 适合吞吐；`SCHED_FIFO`/`SCHED_DEADLINE` 用于硬实时控制线程。
  - `mlockall`、禁用交换、CPU isolation（`isolcpus`/`nohz_full`）是机器人主控常见配方。
- **对 wiki 的映射：** [operating-system-basics](../../wiki/concepts/operating-system-basics.md)、[real-time-control-middleware-guide](../../wiki/queries/real-time-control-middleware-guide.md)

### 3) IETF RFC — 传输与应用层

| RFC | 主题 | 机器人侧用法 |
|-----|------|--------------|
| [RFC 793](https://www.rfc-editor.org/rfc/rfc793) / [9293](https://www.rfc-editor.org/rfc/rfc9293) | TCP | 可靠回传、SSH、模型下载；**不**适合 1 kHz 力矩 |
| [RFC 768](https://www.rfc-editor.org/rfc/rfc768) | UDP | LCM/DDS/自定义传感流；可丢包、低握手 |
| [RFC 9110](https://www.rfc-editor.org/rfc/rfc9110) | HTTP 语义 | 机器人服务 API、遥操作网关、模型仓库 |
| [RFC 1034/1035](https://www.rfc-editor.org/rfc/rfc1035) | DNS | 服务发现前置；边缘离线需本地解析/静态 hosts |
| [RFC 8446](https://www.rfc-editor.org/rfc/rfc8446) | TLS 1.3 | 云端/OTA/遥操作加密；注意握手与证书轮换 |

- **对 wiki 的映射：** [network-protocol-stack](../../wiki/concepts/network-protocol-stack.md)、[udp-multicast-dynamics](../../wiki/formalizations/udp-multicast-dynamics.md)

### 4) 负载均衡（LB）

- **来源：** NGINX / Envoy / Kubernetes Service 文档（[Kubernetes Services](https://kubernetes.io/docs/concepts/services-networking/service/)）。
- **要点：** L4（TCP/UDP）与 L7（HTTP）分流；机器人训练集群与多机仿真常用；真机控制面应避免把实时流量送进 LB。
- **对 wiki 的映射：** [network-protocol-stack](../../wiki/concepts/network-protocol-stack.md)、[container-orchestration-cicd](../../wiki/concepts/container-orchestration-cicd.md)

## 推荐继续阅读（外部）

- man7.org：`sched(7)`、`epoll(7)`、`tcp(7)`、`udp(7)`
- [ROS 2 官方文档](./ros2-official-documentation.md)（DDS 走 UDP，QoS 覆盖可靠性）

## 当前提炼状态

- [x] 摘要与 wiki 映射
