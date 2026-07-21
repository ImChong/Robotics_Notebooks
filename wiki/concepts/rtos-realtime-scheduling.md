---
type: concept
tags: [systems-engineering, rtos, realtime, scheduling, freertos, preempt-rt]
status: complete
updated: 2026-07-21
related:
  - ./operating-system-basics.md
  - ../queries/real-time-control-middleware-guide.md
  - ../formalizations/control-loop-latency-modeling.md
  - ./control-inference-frequency-decoupling.md
  - ../overview/topic-systems-engineering.md
sources:
  - ../../sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md
summary: "实时操作系统与实时调度：硬/软实时、优先级调度、FreeRTOS 与 PREEMPT_RT Linux 在机器人分层中的位置。"
---

# RTOS 与实时调度

## 一句话定义

**RTOS 与实时调度** 保证任务在 **截止时间前** 完成：MCU 侧常用 FreeRTOS/裸机，主控侧常用 PREEMPT_RT Linux + 隔离核与 FIFO 优先级。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RTOS | Real-Time Operating System | 实时操作系统 |
| WCET | Worst-Case Execution Time | 最坏执行时间 |
| RMS | Rate-Monotonic Scheduling | 速率单调调度 |
| IRQ | Interrupt Request | 硬件中断请求 |
| PREEMPT_RT | Preemptive Real-Time (Linux) | Linux 实时抢占补丁 |

## 为什么重要

- 力矩环错过周期会直接表现为抖动或倒地；软实时视觉偶发逾期通常可丢帧。
- 「装了 Ubuntu」≠ 实时；需调度策略、中断与内存配置——见 [实时运控中间件配置指南](../queries/real-time-control-middleware-guide.md)。

## 核心原理

1. **硬实时 vs 软实时**：逾期是否构成系统失败。
2. **优先级调度**：高优先级可抢占；避免优先级反转（优先级继承/天花板）。
3. **可调度性**：利用率界（如 RMS 经典界）+ 实测 WCET。
4. **分层**：驱动器 MCU（μs–100μs）→ 主控 RT 线程（0.5–2 ms）→ 非 RT 感知/规划。

## 工程实践

- MCU：FreeRTOS 任务拆分 FOC、通信、看门狗；中断极短。
- 主控：`SCHED_FIFO`、CPU isolation、`mlockall`、避免在 RT 路径 syscall。
- 测量：循环周期抖动直方图，而不是只看平均频率。
- 与 [频率解耦](./control-inference-frequency-decoupling.md) 配合：推理线程低优先级或异核。

## 局限与风险

- `SCHED_FIFO` 死循环可锁死系统。
- 共享内核的容器/桌面干扰会破坏尾延迟保证。

## 关联页面

- [操作系统基础](./operating-system-basics.md)
- [控制/推理频率解耦](./control-inference-frequency-decoupling.md)
- [实时运控中间件配置指南](../queries/real-time-control-middleware-guide.md)

## 参考来源

- [DDS/RTOS/边云/OTA/安全 FSM 一手资料](../../sources/sites/dds_omg_rtos_edge_ota_safety_primary_refs.md)

## 推荐继续阅读

- FreeRTOS 文档；Linux Foundation Realtime Wiki
