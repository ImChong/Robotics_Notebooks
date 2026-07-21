---
type: concept
tags: [systems-engineering, cache, redis, consistency]
status: complete
updated: 2026-07-21
related:
  - ./database-fundamentals.md
  - ./distributed-systems-basics.md
  - ./model-versioning-ota.md
  - ../overview/topic-systems-engineering.md
sources:
  - ../../sources/sites/systems_engineering_data_distributed_primary_refs.md
summary: "缓存穿透、雪崩、击穿与一致性：机器人服务/模型元数据加速层的经典失效模式与工程对策。"
---

# 缓存一致性陷阱（穿透 / 雪崩 / 击穿 / 一致性）

## 一句话定义

**缓存一致性陷阱** 归纳「加速层」失效时如何打穿数据库或返回脏数据——在机器人模型仓库、配置中心与遥测聚合 API 上反复出现。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TTL | Time To Live | 缓存条目存活时间 |
| CDN | Content Delivery Network | 内容分发网络 |
| KV | Key-Value | 键值存储形态 |
| RPO | Recovery Point Objective | 可接受的数据丢失窗口 |

## 为什么重要

- 边缘网关缓存「当前策略版本 / 地图切片」可降延迟；失效模式处理不当会雪崩到中心库或下发错误权重。
- 与运控共享内存「最新值」不同：HTTP 缓存有 TTL 与一致性语义，必须显式设计。

## 核心原理

| 模式 | 现象 | 对策 |
|------|------|------|
| **穿透** | 查不存在的 key，每次打 DB | 布隆过滤、空值短 TTL、参数校验 |
| **雪崩** | 大量 key 同时过期 | TTL 加抖动、多级缓存、预热 |
| **击穿** | 热点 key 过期瞬间并发重建 | 互斥锁重建、逻辑过期、永不过期+异步刷新 |
| **一致性** | 缓存与源不一致 | cache-aside、以版本仓库为单一真相源 |

## 工程实践

1. **模型元数据**：以 registry 版本号为源；缓存 key 含 digest，禁止「最新」无版本指针长期缓存。
2. **地图/标定**：大对象走对象存储 + CDN；网关只缓存清单。
3. **观测**：缓存命中率、重建延迟、DB QPS 尖峰。

## 局限与风险

- 追求强一致缓存 ≈ 没有缓存；机器人配置变更应以推送/版本号打破缓存。
- 本地进程内缓存与分布式缓存双层时，失效通知必须完整。

## 关联页面

- [数据库基础](./database-fundamentals.md)
- [模型版本管理与 OTA](./model-versioning-ota.md)
- [边缘–云端协同](./edge-cloud-robotics.md)

## 参考来源

- [数据与分布式一手资料](../../sources/sites/systems_engineering_data_distributed_primary_refs.md)

## 推荐继续阅读

- Redis Caching patterns 文档
