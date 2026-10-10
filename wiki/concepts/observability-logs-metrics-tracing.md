---
type: concept
tags: [systems-engineering, observability, logging, metrics, tracing, opentelemetry]
status: complete
updated: 2026-10-10
related:
  - ./container-orchestration-cicd.md
  - ./operating-system-basics.md
  - ../formalizations/control-loop-latency-modeling.md
  - ./swinging-door-trending-compression.md
  - ../overview/hub-systems-engineering.md
sources:
  - ../../sources/sites/systems_engineering_deploy_obs_security_primary_refs.md
  - ../../sources/sites/aveva_pi_swinging_door_compression.md
summary: "可观测性（日志、Metrics、Tracing）：云边服务用 OpenTelemetry 三支柱；运控环路用周期/抖动直方图，避免同步埋点拖垮实时性。"
---

# 可观测性（Logs / Metrics / Tracing）

## 一句话定义

**可观测性** 让系统在故障时仍能回答「发生了什么、哪里慢、因果链如何」——对云边服务用三支柱，对硬实时环路用轻量指标。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SLI | Service Level Indicator | 服务水平指标 |
| SLO | Service Level Objective | 服务水平目标 |
| OTel | OpenTelemetry | 遥测数据采集标准 |
| RED | Rate Errors Duration | 请求类服务常用指标集 |
| USE | Utilization Saturation Errors | 资源类指标集 |

## 为什么重要

- 无指标时，「偶发抽搐」无法区分总线、调度还是策略。
- 分布式训练与边云 API 需要 tracing 定位跨服务延迟。

## 核心原理

| 支柱 | 内容 | 机器人用法 |
|------|------|------------|
| **Logs** | 事件叙述 | 异步落盘；含 `robot_id`/版本 |
| **Metrics** | 数值时间序列 | 控制频率、deadline miss、CAN 总线利用率、GPU util |
| **Tracing** | 跨服务因果 | 云边 RPC、数据集管道；非 1 kHz 环 |

## 工程实践

1. 运控进程导出 **周期时间直方图** 与 miss 计数（对齐 [延迟建模](../formalizations/control-loop-latency-modeling.md)）。
2. 日志限速与采样；崩溃用环形缓冲。
3. 云服务跟 RED；节点跟 USE。
4. 告警基于 SLO，而非「有 error log 就叫人」。
5. **队级/地理大屏（可选前端层）：** Metrics 聚合后可在浏览器用 Three.js + 图表库做运营视图；参考样板 [sc-datav](../entities/sc-datav.md)（静态 demo，需自接 API/OTel 查询，非内置后端）。
6. **历史遥测压缩：** 若需长期归档，可在非实时的存储副本上使用 [Swinging Door Trending](./swinging-door-trending-compression.md) 这类有误差界的有损压缩；保留原始控制、安全和故障诊断数据，避免在硬实时环中同步压缩或写盘。

## 局限与风险

- 在 `SCHED_FIFO` 线程里同步打 span/写磁盘 → 自我制造抖动。
- 高基数标签（每关节每毫秒）会撑爆时序库。

## 关联页面

- [操作系统基础](./operating-system-basics.md)
- [容器编排与 CI/CD](./container-orchestration-cicd.md)
- [系统工程知识链](../overview/hub-systems-engineering.md)
- [sc-datav（Three.js 数据可视化大屏）](../entities/sc-datav.md)

## 参考来源

- [部署可观测安全一手资料](../../sources/sites/systems_engineering_deploy_obs_security_primary_refs.md)
- [AVEVA PI 的 Swinging Door 压缩资料](../../sources/sites/aveva_pi_swinging_door_compression.md)

## 推荐继续阅读

- OpenTelemetry 文档：<https://opentelemetry.io/docs/>
