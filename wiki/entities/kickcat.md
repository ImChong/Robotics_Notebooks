---
type: entity
project_id: kickcat
code: "https://github.com/leducp/KickCAT"
tags: [ethercat, fieldbus, open-source, master, embedded, realtime]
status: draft
updated: 2026-10-14
related:
  - ./soem.md
  - ./ethercat-virtual-slave-sim.md
  - ../concepts/ethercat-protocol.md
  - ../overview/ethercat-opensource-libs-wechat.md
sources:
  - ../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md
  - ../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md
summary: "KickCAT：C++ 面向对象的 EtherCAT 主站库，自带软件模拟从站，适合上位机开发调试。"
---

# KickCAT

## 一句话定义

C++ 面向对象的 EtherCAT 主站库，自带软件模拟从站，适合上位机开发调试。（分类：EtherCAT 主站；语言/形态：C++）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EtherCAT | Ethernet for Control Automation Technology | 工业以太网现场总线 |
| ESC | EtherCAT Slave Controller | 从站硬件芯片（如 ET1100 / LAN9252 / ET1815） |
| CoE | CANopen over EtherCAT | 最常用的 EtherCAT 应用层协议 |

## 为什么重要

- 在 [EtherCAT 开源库清单](../overview/ethercat-opensource-libs-wechat.md) 中属于「主站」一栏，是机器人关节总线工具链的一环。
- C++ 上位机项目希望用现代 C++ 接口、并想脱离硬件做主站逻辑联调时的备选。

## 核心原理

文章对该项目的描述（以下均为文章声明，未独立核查）：

- 语言：C++，面向对象 API。
- 文章称自带软件模拟从站，便于在无硬件时调试上位机。
- 文章称项目活跃度一般。

## 工程实践

- 适用场景：C++ 上位机项目希望用现代 C++ 接口、并想脱离硬件做主站逻辑联调时的备选。
- 官方地址：<https://github.com/leducp/KickCAT>（文章给出，本次未独立打开核查）
- 协议背景见 [EtherCAT 协议基础](../concepts/ethercat-protocol.md)；主站实时调优见 [主站优化指南](../queries/ethercat-master-optimization.md)。

## 局限与风险

- **开源状态：** 文章声明为公开仓库；**本次会话无法访问该仓库，开源程度/许可证未独立核查**，待后续按 ingest 步骤 2.5 补查。
- **推测/待核查：** 文章中「EtherCAT Master Sim」被注明 KickCAT 自带，见 [EtherCAT 虚拟从站仿真](./ethercat-virtual-slave-sim.md)；二者关系以仓库为准（推测，未核查）。
- EtherCAT 协议本身不开源，协议栈代码开源，ESC 硬件芯片需单独购买（文章避坑提醒）。

## 关联页面

- [soem](./soem.md)
- [ethercat-virtual-slave-sim](./ethercat-virtual-slave-sim.md)
- [ethercat-protocol](../concepts/ethercat-protocol.md)
- [ethercat-opensource-libs-wechat](../overview/ethercat-opensource-libs-wechat.md)

## 参考来源

- [sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md](../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md)
- [sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md](../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md)

## 推荐继续阅读

- EtherCAT Technology Group：<https://www.ethercat.org/>
- 文章原文：<https://mp.weixin.qq.com/s/C34QLsVb7Sd-GlvPvQlGKQ>
