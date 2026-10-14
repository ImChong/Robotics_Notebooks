---
type: entity
project_id: igh-ethercat-master
code: "https://gitlab.com/etherlab.org/ethercat"
tags: [ethercat, fieldbus, open-source, master, embedded, realtime]
status: draft
updated: 2026-10-14
related:
  - ./soem.md
  - ../queries/ethercat-master-optimization.md
  - ../concepts/ethercat-protocol.md
  - ../overview/ethercat-opensource-libs-wechat.md
sources:
  - ../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md
  - ../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md
summary: "IgH EtherCAT Master (EtherLab)：以 Linux 内核模块形式实现的开源 EtherCAT 主站，文章将其定位为「高性能硬实时」首选。"
---

# IgH EtherCAT Master (EtherLab)

## 一句话定义

以 Linux 内核模块形式实现的开源 EtherCAT 主站，文章将其定位为「高性能硬实时」首选。（分类：EtherCAT 主站；语言/形态：C，Linux 内核模块）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EtherCAT | Ethernet for Control Automation Technology | 工业以太网现场总线 |
| DC | Distributed Clock | 从站分布式时钟同步 |
| PREEMPT_RT | Linux real-time preemption patch | Linux 实时内核路径 |

## 为什么重要

- 在 [EtherCAT 开源库清单](../overview/ethercat-opensource-libs-wechat.md) 中属于「主站」一栏，是机器人关节总线工具链的一环。
- 微秒级同步、Linux 实时工控主机；与用户态 [SOEM](./soem.md) 构成主站选型对照。

## 核心原理

文章对该项目的描述（以下均为文章声明，未独立核查）：

- 语言：C；运行形态为 Linux 内核模块（内核态）。
- 文章称其实时性最强、DC 同步精度极高，在工业 CNC 与机器人中大量生产使用。
- 文章称仅支持 Linux（Preempt-RT / Xenomai），移植难度高。
- 文章选型句：Linux 硬实时、机器人/CNC 高同步 → IgH EtherLab；高同步伺服场景优先 IgH（SOEM 的 DC 精度弱于它）。

## 工程实践

- 适用场景：微秒级同步、Linux 实时工控主机；与用户态 [SOEM](./soem.md) 构成主站选型对照。
- 官方地址：<https://gitlab.com/etherlab.org/ethercat>（文章给出，本次未独立打开核查）
- 协议背景见 [EtherCAT 协议基础](../concepts/ethercat-protocol.md)；主站实时调优见 [主站优化指南](../queries/ethercat-master-optimization.md)。

## 局限与风险

- **开源状态：** 文章声明为公开仓库；**本次会话无法访问该仓库，开源程度/许可证未独立核查**，待后续按 ingest 步骤 2.5 补查。
- **推测/待核查：** 内核态主站通常需要专用网卡驱动并随内核版本维护，这是移植成本的主要来源（推测，来自对内核模块形态的一般理解，非文章内容）。
- EtherCAT 协议本身不开源，协议栈代码开源，ESC 硬件芯片需单独购买（文章避坑提醒）。

## 关联页面

- [soem](./soem.md)
- [ethercat-master-optimization](../queries/ethercat-master-optimization.md)
- [ethercat-protocol](../concepts/ethercat-protocol.md)
- [ethercat-opensource-libs-wechat](../overview/ethercat-opensource-libs-wechat.md)

## 参考来源

- [sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md](../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md)
- [sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md](../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md)

## 推荐继续阅读

- EtherCAT Technology Group：<https://www.ethercat.org/>
- 文章原文：<https://mp.weixin.qq.com/s/C34QLsVb7Sd-GlvPvQlGKQ>
