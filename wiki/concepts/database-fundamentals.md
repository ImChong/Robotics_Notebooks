---
type: concept
tags: [systems-engineering, database, transactions, indexing, replication]
status: complete
updated: 2026-07-21
related:
  - ./cache-consistency-pitfalls.md
  - ./message-queue-reliability.md
  - ./distributed-systems-basics.md
  - ../overview/topic-systems-engineering.md
sources:
  - ../../sources/sites/systems_engineering_data_distributed_primary_refs.md
summary: "数据库基础（索引、事务、锁、隔离级别、复制、分片）：服务机器人实验元数据、遥测与队级数据面，而非 1 kHz 运控环。"
---

# 数据库基础（索引 / 事务 / 锁 / 隔离级别 / 复制 / 分片）

## 一句话定义

**数据库基础** 说明如何在机器人研发数据面里 **正确存、正确并发写、正确扩展读**，并明确它不属于力矩闭环路径。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ACID | Atomicity Consistency Isolation Durability | 事务四性质 |
| MVCC | Multi-Version Concurrency Control | 多版本并发控制 |
| WAL | Write-Ahead Logging | 先写日志再落库 |
| OLTP | Online Transaction Processing | 在线事务处理 |
| OLAP | Online Analytical Processing | 分析型查询 |

## 为什么重要

- 实验配置、数据集谱系、遥测入库、多机器人任务状态都落在 DB。
- 隔离级别选错会导致「两个训练 Job 改同一实验记录」的幽灵数据；分片不当会让跨机器人联合查询极慢。

## 核心原理

1. **索引**：B-Tree / Hash 等加速查找；高频写入的原始传感流更宜时序库或对象存储 + 元数据索引。
2. **事务与锁**：原子提交；行锁/表锁权衡并发。
3. **隔离级别**：Read Committed 常见默认；需要可重复读或可串行化时显式声明。
4. **复制**：主从异步换读扩展；注意复制延迟下的「读己之写」。
5. **分片**：按 `robot_id` / 时间范围切分；跨分片事务尽量避免。

## 工程实践

- **元数据**（实验、模型版本）：强一致 OLTP（如 PostgreSQL）。
- **高频遥测**：写入消息队列或时序库，DB 只存降采样与标注。
- **迁移**：schema 变更进 CI；禁止生产手工改表。
- **备份**：逻辑备份 + 时间点恢复演练。

## 局限与风险

- 把关节状态以行插入 OLTP 每毫秒一次——写放大会拖垮整库。
- 「开了复制 = 不丢数据」——异步复制在主挂时可能丢最近事务。

## 关联页面

- [缓存一致性陷阱](./cache-consistency-pitfalls.md)
- [消息队列可靠性](./message-queue-reliability.md)
- [分布式系统基础](./distributed-systems-basics.md)

## 参考来源

- [数据与分布式一手资料](../../sources/sites/systems_engineering_data_distributed_primary_refs.md)

## 推荐继续阅读

- PostgreSQL Transaction Isolation 文档
