---
type: concept
tags: [systems-engineering, messaging, queue, idempotency, kafka]
status: complete
updated: 2026-07-21
related:
  - ./distributed-systems-basics.md
  - ./dds-communication.md
  - ./lcm-basics.md
  - ./database-fundamentals.md
  - ../overview/topic-systems-engineering.md
sources:
  - ../../sources/sites/systems_engineering_data_distributed_primary_refs.md
summary: "消息系统可靠性（队列、重复消费、顺序、幂等）：云边任务与遥测管道语义；对比 DDS/LCM 的实时发布订阅。"
---

# 消息队列可靠性（队列 / 重复消费 / 顺序 / 幂等）

## 一句话定义

**消息队列可靠性** 处理「异步投递」下的 **至少一次、重复、乱序** 现实，要求消费者 **幂等**——适用于任务调度与遥测，不替代实时中间件。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MQ | Message Queue | 消息队列 |
| ACK | Acknowledgement | 消费确认 |
| FIFO | First In First Out | 先进先出顺序 |
| EOS | Exactly-Once Semantics | 精确一次语义（常需事务/幂等配合） |
| Pub/Sub | Publish / Subscribe | 发布订阅 |

## 为什么重要

- 多机器人任务下发、数据集导入、告警聚合常用 Kafka/RabbitMQ/云队列。
- 默认 **至少一次** 会重复投递；若「开始标定」「扣减库存式」副作用不幂等，会产生双执行事故。

## 核心原理

1. **队列 vs 总线**：MQ 强调持久化与积压；[DDS](./dds-communication.md)/[LCM](./lcm-basics.md) 强调实时最新值。
2. **重复消费**：超时重投、消费者崩溃重启都会重复 → 用业务幂等键。
3. **顺序**：通常仅分区内有序；需要全局顺序则单分区，吞吐下降。
4. **幂等**：同一 `message_id` 执行多次结果相同（或第二次成 no-op）。

## 工程实践

- 消息体带 `robot_id`、`command_id`、`schema_version`。
- 消费侧先写去重表再执行副作用；或利用 DB 唯一约束。
- 死信队列承接毒消息；告警而不是无限重试。
- **禁止** 用 MQ 传 500 Hz 关节指令。

## 局限与风险

- 「开启 exactly-once」仍可能在应用层重复——端到端幂等不可省。
- 积压时延迟无界，不能当实时 deadline 机制。

## 关联页面

- [分布式系统基础](./distributed-systems-basics.md)
- [DDS 通信机制](./dds-communication.md)
- [可观测性](./observability-logs-metrics-tracing.md)

## 参考来源

- [数据与分布式一手资料](../../sources/sites/systems_engineering_data_distributed_primary_refs.md)

## 推荐继续阅读

- Apache Kafka Documentation — delivery semantics
