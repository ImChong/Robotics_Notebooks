---
type: entity
project_id: cherryecat
tags: [ethercat, fieldbus, open-source, master, embedded, realtime]
status: draft
updated: 2026-10-14
related:
  - ./igh-ethercat-master.md
  - ./soem.md
  - ../concepts/ethercat-protocol.md
  - ../overview/ethercat-opensource-libs-wechat.md
sources:
  - ../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md
  - ../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md
summary: "CherryECAT：面向 MCU + RTOS 裁剪的轻量 EtherCAT 主站，文章称基于 IgH 改造。"
---

# CherryECAT

## 一句话定义

面向 MCU + RTOS 裁剪的轻量 EtherCAT 主站，文章称基于 IgH 改造。（分类：EtherCAT 主站；语言/形态：C，基于 IgH 改造）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EtherCAT | Ethernet for Control Automation Technology | 工业以太网现场总线 |
| RTOS | Real-Time Operating System | FreeRTOS / RT-Thread 等 |
| DC | Distributed Clock | 从站分布式时钟同步 |

## 为什么重要

- 在 [EtherCAT 开源库清单](../overview/ethercat-opensource-libs-wechat.md) 中属于「主站」一栏，是机器人关节总线工具链的一环。
- 想在 MCU 上直接做 EtherCAT 主站（无 Linux）时的候选；与 [IgH](./igh-ethercat-master.md) 的血缘关系见上。

## 核心原理

文章对该项目的描述（以下均为文章声明，未独立核查）：

- 语言：C；文章称基于 IgH 改造。
- 文章称专为 MCU RTOS（FreeRTOS / RT-Thread）裁剪，轻量化。
- 文章仓库栏仅写「RT-Thread 社区」，未给 URL。

## 工程实践

- 适用场景：想在 MCU 上直接做 EtherCAT 主站（无 Linux）时的候选；与 [IgH](./igh-ethercat-master.md) 的血缘关系见上。
- 官方地址：文章未给出，本页不臆造 URL
- 协议背景见 [EtherCAT 协议基础](../concepts/ethercat-protocol.md)；主站实时调优见 [主站优化指南](../queries/ethercat-master-optimization.md)。

## 局限与风险

- **开源状态：** **入库日（2026-10-14）未能确认官方仓库与开源程度**，需补查；本页仅编译文章线索。
- **推测/待核查：** 官方仓库地址文中缺失，本页不臆造 URL；入库日（2026-10-14）未能核实具体仓库、许可证与开源程度，需补查。
- **推测/待核查：** 因基于 IgH，许可证很可能受 IgH 许可约束（推测，未核查）。
- EtherCAT 协议本身不开源，协议栈代码开源，ESC 硬件芯片需单独购买（文章避坑提醒）。

## 关联页面

- [igh-ethercat-master](./igh-ethercat-master.md)
- [soem](./soem.md)
- [ethercat-protocol](../concepts/ethercat-protocol.md)
- [ethercat-opensource-libs-wechat](../overview/ethercat-opensource-libs-wechat.md)

## 参考来源

- [sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md](../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md)
- [sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md](../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md)

## 推荐继续阅读

- EtherCAT Technology Group：<https://www.ethercat.org/>
- 文章原文：<https://mp.weixin.qq.com/s/C34QLsVb7Sd-GlvPvQlGKQ>
