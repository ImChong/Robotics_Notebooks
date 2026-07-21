# 数据库、缓存、消息与分布式系统一手资料索引

> 来源归档（ingest）

- **标题：** 索引/事务/隔离、缓存失效模式、消息可靠性与 CAP/选主/一致性一手依据
- **类型：** standard / paper / official docs（合集）
- **入库日期：** 2026-07-21
- **一句话说明：** 覆盖机器人云端数据面（实验追踪、数据集、遥测、多机协同）所需的数据与分布式基础，避免把「后端八股」原样塞进运控环。
- **沉淀到 wiki：** 是 → [database-fundamentals](../../wiki/concepts/database-fundamentals.md)、[cache-consistency-pitfalls](../../wiki/concepts/cache-consistency-pitfalls.md)、[message-queue-reliability](../../wiki/concepts/message-queue-reliability.md)、[distributed-systems-basics](../../wiki/concepts/distributed-systems-basics.md)

## 为什么值得保留

- 人形研发栈同时有 **实时控制面** 与 **云/边数据面**：W&B、数据集仓、遥测入库、多机器人任务调度都属于后者。
- 混用两套语义是常见故障源：把「至少一次投递」的消息队列当成 1 kHz 关节总线，或把强一致事务期望套到 DDS BestEffort 话题上。

## 核心摘录

### 1) 数据库：索引、事务、锁、隔离、复制、分片

- **来源：**
  - PostgreSQL 文档：[Indexes](https://www.postgresql.org/docs/current/indexes.html)、[Transaction Isolation](https://www.postgresql.org/docs/current/transaction-iso.html)、[Replication](https://www.postgresql.org/docs/current/high-availability.html)
  - ANSI SQL 隔离级别语义（Read Uncommitted / Committed / Repeatable Read / Serializable）
  - Gray & Reuter《Transaction Processing》思想：ACID、锁与日志
- **要点：**
  - **索引**降低点查/范围查成本，但写入放大；遥测时序更常选时序库或列存。
  - **事务 + 隔离级别**决定并发异常（脏读/不可重复读/幻读）；实验元数据写用可串行化或至少 RC。
  - **复制**换读扩展与灾备；**分片**换写扩展，跨分片事务昂贵——机器人多机日志宜按 robot_id/时间分区。
- **对 wiki 的映射：** [database-fundamentals](../../wiki/concepts/database-fundamentals.md)

### 2) 缓存：穿透、雪崩、击穿、一致性

- **来源：** Redis 文档 [Caching patterns](https://redis.io/docs/latest/develop/use/patterns/)；业界常见失效模式综述（cache-aside / write-through）。
- **要点：**
  - **穿透**：查不存在的 key 打穿到 DB → 布隆过滤 / 空值短 TTL。
  - **雪崩**：大量 key 同时过期 → 抖动 TTL / 分层缓存。
  - **击穿**：热点 key 失效瞬间并发重建 → 互斥重建 / 逻辑过期。
  - **一致性**：缓存与 DB 双写无银弹；机器人模型元数据宜「以版本仓库为源」，缓存只加速。
- **对 wiki 的映射：** [cache-consistency-pitfalls](../../wiki/concepts/cache-consistency-pitfalls.md)

### 3) 消息系统：队列、重复消费、顺序、幂等

- **来源：** [Apache Kafka 文档](https://kafka.apache.org/documentation/)（at-least-once / exactly-once 语义）；[AMQP 0-9-1](https://www.rabbitmq.com/docs/amqp)（RabbitMQ）。
- **要点：**
  - 默认多为 **至少一次**：消费者必须 **幂等**（按 message_id / 业务键去重）。
  - **顺序**通常只能保证分区/队列内；跨分区全局顺序昂贵。
  - 与 DDS/LCM 对比：消息队列强调持久化与积压；运控中间件强调最新值与低延迟。
- **对 wiki 的映射：** [message-queue-reliability](../../wiki/concepts/message-queue-reliability.md)、[dds-communication](../../wiki/concepts/dds-communication.md)

### 4) 分布式：CAP、选主、一致性、超时、重试

- **来源：**
  - Brewer CAP 猜想 / Gilbert–Lynch 形式化（PODC 2002）
  - Ongaro & Ousterhout, *[In Search of an Understandable Consensus Algorithm (Raft)](https://raft.github.io/raft.pdf)*
  - AWS / Google SRE 实践：超时、退避重试、幂等、熔断
- **要点：**
  - **CAP**：分区时在一致性与可用性间取舍；机器人机载常选「本地可用 + 最终一致上云」。
  - **选主**：Raft/etcd/K8s 控制面；机器人 onboard 安全 FSM 的「主控」应是确定性本地逻辑，而非网络选主。
  - **超时/重试**：无界重试会放大故障；须抖动退避 + 幂等 + 最大尝试次数。
- **对 wiki 的映射：** [distributed-systems-basics](../../wiki/concepts/distributed-systems-basics.md)

## 当前提炼状态

- [x] 摘要与 wiki 映射
