---
type: concept
tags: [systems-engineering, os, linux, scheduling, memory, deployment]
status: complete
updated: 2026-07-21
related:
  - ./rtos-realtime-scheduling.md
  - ./network-protocol-stack.md
  - ../queries/real-time-control-middleware-guide.md
  - ../overview/topic-systems-engineering.md
  - ../formalizations/control-loop-latency-modeling.md
sources:
  - ../../sources/sites/systems_engineering_os_network_primary_refs.md
summary: "操作系统基础（进程、线程、内存、文件系统、调度）：机器人主控 Linux 上隔离运控与日志 I/O、理解抖动来源的最小知识集。"
---

# 操作系统基础（进程 / 线程 / 内存 / 文件系统 / 调度）

## 一句话定义

**操作系统基础** 解释主控板上「谁在跑、何时被抢占、内存如何映射、磁盘 I/O 如何饿死控制线程」——是诊断真机抖动时最先翻的一层。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OS | Operating System | 管理 CPU、内存、设备与进程的系统软件 |
| IPC | Inter-Process Communication | 进程间通信（共享内存、套接字等） |
| TLB | Translation Lookaside Buffer | 地址翻译缓存，未命中会拉长延迟 |
| CFS | Completely Fair Scheduler | Linux 默认公平调度器 |
| FS | File System | 文件系统，持久化与同步写会影响延迟 |

## 为什么重要

- 仿真里「完美策略」上真机抽搐，常见根因是 **调度抖动 / 缺页 / 同步写盘**，见 [控制环路延迟建模](../formalizations/control-loop-latency-modeling.md) 中的 $T_{\text{sched}}$。
- 进程边界决定崩溃隔离；线程边界决定能否把 1 kHz 控制与视觉推理绑在不同优先级。

## 核心原理

| 概念 | 机制 | 机器人侧含义 |
|------|------|--------------|
| **进程** | 独立地址空间与 FD 表 | 感知、控制、日志可分进程，故障隔离 |
| **线程** | 共享地址空间的执行流 | 同进程内传感读、策略前向、发布可并行 |
| **虚拟内存** | 页表 + 换页 | 实时路径应 `mlock`、避免冷分配 |
| **文件系统** | 缓存、日志、`fsync` | rosbag/日志同步写可造成毫秒级尖刺 |
| **调度** | 时间片与优先级 | 运控用 `SCHED_FIFO`；普通用 CFS |

## 工程实践

1. **控制线程**：固定 CPU、提高优先级、关闭交换；细节见 [实时运控中间件配置指南](../queries/real-time-control-middleware-guide.md)。
2. **日志与 bag**：异步队列 + 独立核/进程，禁止在力矩环内 `fprintf`/`fsync`。
3. **内存**：启动时预分配；推理引擎选固定 workspace，避免运行时 `malloc` 风暴。
4. **观察指标**：上下文切换次数、软中断、major page fault、磁盘 await。

## 局限与风险

- 把「多线程」等同于「实时」——无优先级与隔离时只会更吵。
- 在 NFS/FUSE 上直接写高频日志会引入不可控网络延迟。
- 容器默认共享宿主机调度；GPU 训练 Pod ≠ 硬实时环境。

## 关联页面

- [RTOS 与实时调度](./rtos-realtime-scheduling.md)
- [网络协议栈基础](./network-protocol-stack.md)
- [系统工程专题](../overview/topic-systems-engineering.md)

## 参考来源

- [OS 与网络一手资料](../../sources/sites/systems_engineering_os_network_primary_refs.md)

## 推荐继续阅读

- OSTEP 免费教材：<https://ostep.org/>
