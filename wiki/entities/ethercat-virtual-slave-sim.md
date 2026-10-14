---
type: entity
project_id: ethercat-virtual-slave-sim
tags: [ethercat, fieldbus, open-source, sim, embedded, realtime, independent-maintainer]
status: draft
updated: 2026-10-14
related:
  - ./kickcat.md
  - ./soem.md
  - ../concepts/ethercat-protocol.md
  - ../overview/ethercat-opensource-libs-wechat.md
sources:
  - ../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md
  - ../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md
summary: "EtherCAT 虚拟从站仿真（EtherCAT Master Sim / ethercat-master-sim）：文章「仿真/调试工具」栏的两条同类条目：无需真实硬件、在 PC 上调试主站代码的虚拟从站。"
---

# EtherCAT 虚拟从站仿真（EtherCAT Master Sim / ethercat-master-sim）

## 一句话定义

文章「仿真/调试工具」栏的两条同类条目：无需真实硬件、在 PC 上调试主站代码的虚拟从站。（分类：EtherCAT 仿真/调试；语言/形态：软件虚拟从站）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EtherCAT | Ethernet for Control Automation Technology | 工业以太网现场总线 |
| DC | Distributed Clock | 从站分布式时钟同步 |
| ESC | EtherCAT Slave Controller | 从站硬件芯片（如 ET1100 / LAN9252 / ET1815） |

## 为什么重要

- 在 [EtherCAT 开源库清单](../overview/ethercat-opensource-libs-wechat.md) 中属于「仿真/调试」一栏，是机器人关节总线工具链的一环。
- 没有硬件时先用 [KickCAT](./kickcat.md) 的模拟从站联调主站逻辑；上真机后仍需用实物验证 DC 与抖动。

## 核心原理

文章对该项目的描述（以下均为文章声明，未独立核查）：

- 文章条目一「EtherCAT Master Sim」：软件虚拟从站，不需要真实硬件，电脑直接调试主站代码，并注明 KickCAT 自带。
- 文章条目二「ethercat-master-sim」：虚拟 EtherCAT 从站，用于 PC 端开发测试。
- 两条描述功能重合、均未给仓库地址。

## 工程实践

- 适用场景：没有硬件时先用 [KickCAT](./kickcat.md) 的模拟从站联调主站逻辑；上真机后仍需用实物验证 DC 与抖动。
- 官方地址：文章未给出，本页不臆造 URL
- 协议背景见 [EtherCAT 协议基础](../concepts/ethercat-protocol.md)；主站实时调优见 [主站优化指南](../queries/ethercat-master-optimization.md)。

## 局限与风险

- **开源状态：** **入库日（2026-10-14）未能确认官方仓库与开源程度**，需补查；本页仅编译文章线索。
- **推测/待核查：** 两条名称近似、描述重合，我判断为同一类工具的两种写法，故合并为一个节点以避免重复；是否实为同一项目未核查。
- **推测/待核查：** 入库日未能定位到唯一官方仓库，本页登记「方法线索」，不是已验证的开源项目。
- EtherCAT 协议本身不开源，协议栈代码开源，ESC 硬件芯片需单独购买（文章避坑提醒）。

## 关联页面

- [kickcat](./kickcat.md)
- [soem](./soem.md)
- [ethercat-protocol](../concepts/ethercat-protocol.md)
- [ethercat-opensource-libs-wechat](../overview/ethercat-opensource-libs-wechat.md)

## 参考来源

- [sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md](../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md)
- [sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md](../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md)

## 推荐继续阅读

- EtherCAT Technology Group：<https://www.ethercat.org/>
- 文章原文：<https://mp.weixin.qq.com/s/C34QLsVb7Sd-GlvPvQlGKQ>
