---
type: overview
tags: [ethercat, fieldbus, open-source, curated-index, wechat-curator, realtime]
status: draft
updated: 2026-10-14
related:
  - ../entities/soem.md
  - ../entities/igh-ethercat-master.md
  - ../entities/kickcat.md
  - ../entities/ethercrab.md
  - ../entities/cherryecat.md
  - ../entities/libethercat-dlr.md
  - ../entities/soes.md
  - ../entities/beckhoff-ssc.md
  - ../entities/pysoem.md
  - ../entities/soem-qt.md
  - ../entities/ros2-ethercat.md
  - ../entities/ecat-servo.md
  - ../entities/opensml.md
  - ../entities/ethercat-virtual-slave-sim.md
  - ../concepts/ethercat-protocol.md
  - ../queries/ethercat-master-optimization.md
sources:
  - ../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md
  - ../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md
summary: "微信公众号「Qt历险记」EtherCAT 开源库清单：主站/从站/绑定/CiA402/仿真共 14 个独立详情节点（含已有 SOEM），附文章选型句与未核查项标注。"
---

# EtherCAT 开源库大全（微信策展）— 索引

## 一句话定义

把 [微信文章](https://mp.weixin.qq.com/s/C34QLsVb7Sd-GlvPvQlGKQ) 中的 EtherCAT 开源库拆成独立、不重复的 wiki 详情节点，并整理文章给出的选型建议。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EtherCAT | Ethernet for Control Automation Technology | 工业以太网现场总线 |
| ESC | EtherCAT Slave Controller | 从站硬件芯片 |
| CiA402 | CAN in Automation 402 drive profile | 伺服驱动器 Profile |
| DC | Distributed Clock | 分布式时钟同步 |

## 为什么重要

人形/机械臂关节总线选 EtherCAT 后，主站、从站、上位机封装与 CiA402 伺服各层都要选开源件；本页给出清单与文章选型，协议基础见 [EtherCAT 协议基础](../concepts/ethercat-protocol.md)。

## 核心原理：14 项一览

| 分区 | 详情节点 | 语言 / 形态（文章） |
|------|----------|---------------------|
| 主站 | [SOEM](../entities/soem.md) | C，用户态 |
| 主站 | [IgH EtherCAT Master](../entities/igh-ethercat-master.md) | C，内核模块 |
| 主站 | [KickCAT](../entities/kickcat.md) | C++ |
| 主站 | [EtherCrab](../entities/ethercrab.md) | Rust |
| 主站 | [CherryECAT](../entities/cherryecat.md) | C，基于 IgH，MCU RTOS |
| 主站 | [libethercat（DLR）](../entities/libethercat-dlr.md) | C 库，科研 |
| 从站 | [SOES](../entities/soes.md) | C |
| 从站 | [Beckhoff SSC](../entities/beckhoff-ssc.md) | 官方栈，ETG 申请 |
| 绑定 | [pysoem](../entities/pysoem.md) | Python |
| 绑定 | [soem-qt](../entities/soem-qt.md) | Qt C++ |
| 绑定 | [ROS2-ethercat](../entities/ros2-ethercat.md) | ROS 2 |
| 伺服 | [ecat_servo](../entities/ecat-servo.md) | CiA402 状态机 |
| 伺服 | [OpenSML](../entities/opensml.md) | SoftMotion |
| 仿真 | [EtherCAT 虚拟从站仿真](../entities/ethercat-virtual-slave-sim.md) | 文中两条合并为一个节点 |

## 工程实践：文章选型句

- Qt 上位机、Windows/嵌入式 Linux、快速开发 → SOEM。
- Linux 硬实时、机器人/CNC 高同步 → IgH EtherLab。
- 自制硬件从站 → SOES + LAN9252。
- Rust 项目 → Ethercrab。
- 避坑（文章）：SOEM 用户态，DC 精度弱于 IgH；协议本身不开源；Windows 普通网卡实时性差。

## 局限与风险

- 本次入库**无法访问文中 GitHub 仓库**，开源程度/许可证均未独立核查；详见 [来源归档](../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md)。
- soem-qt、ROS2-ethercat、虚拟从站仿真、CherryECAT、libethercat、pysoem 文中无仓库地址，节点仅编译文章线索（我推测其中几项是泛称，未核查）。
- 文章为转载式清单，性能「微秒级」之类说法无实测数据。

## 关联页面

- [EtherCAT 协议基础](../concepts/ethercat-protocol.md)
- [EtherCAT 主站优化指南](../queries/ethercat-master-optimization.md)
- [CAN vs EtherCAT 关节总线选型](../comparisons/can-vs-ethercat-joint-bus.md)

## 参考来源

- [sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md](../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md)
- [sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md](../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md)

## 推荐继续阅读

- EtherCAT Technology Group：<https://www.ethercat.org/>
