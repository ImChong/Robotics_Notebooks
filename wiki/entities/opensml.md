---
type: entity
project_id: opensml
code: "https://github.com/feecat/OpenSML"
tags: [ethercat, fieldbus, open-source, servo, embedded, realtime]
status: draft
updated: 2026-10-14
related:
  - ./ecat-servo.md
  - ./soem.md
  - ../concepts/ethercat-protocol.md
  - ../overview/ethercat-opensource-libs-wechat.md
sources:
  - ../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md
  - ../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md
summary: "OpenSML：适配 CiA402 的轻量 SoftMotion 运动控制库，提供点位与插补。"
---

# OpenSML

## 一句话定义

适配 CiA402 的轻量 SoftMotion 运动控制库，提供点位与插补。（分类：EtherCAT 伺服 CiA402 配套；语言/形态：SoftMotion 轻量运动控制库）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CiA402 | CAN in Automation 402 drive profile | 伺服驱动器状态机与运动模式标准 |
| CoE | CANopen over EtherCAT | 最常用的 EtherCAT 应用层协议 |
| EtherCAT | Ethernet for Control Automation Technology | 工业以太网现场总线 |

## 为什么重要

- 在 [EtherCAT 开源库清单](../overview/ethercat-opensource-libs-wechat.md) 中属于「伺服 CiA402 配套」一栏，是机器人关节总线工具链的一环。
- 想在 CNC/多轴场景自带轨迹插补而非依赖商用运动控制器时的参考。

## 核心原理

文章对该项目的描述（以下均为文章声明，未独立核查）：

- 文章称为 SoftMotion 轻量运动控制库，适配 CiA402，支持点位、插补。
- 仓库地址由文章给出。

## 工程实践

- 适用场景：想在 CNC/多轴场景自带轨迹插补而非依赖商用运动控制器时的参考。
- 官方地址：<https://github.com/feecat/OpenSML>（文章给出，本次未独立打开核查）
- 协议背景见 [EtherCAT 协议基础](../concepts/ethercat-protocol.md)；主站实时调优见 [主站优化指南](../queries/ethercat-master-optimization.md)。

## 局限与风险

- **开源状态：** 文章声明为公开仓库；**本次会话无法访问该仓库，开源程度/许可证未独立核查**，待后续按 ingest 步骤 2.5 补查。
- **推测/待核查：** 未核查其语言、许可证与依赖的主站（是否基于 SOEM/IgH）。
- EtherCAT 协议本身不开源，协议栈代码开源，ESC 硬件芯片需单独购买（文章避坑提醒）。

## 关联页面

- [ecat-servo](./ecat-servo.md)
- [soem](./soem.md)
- [ethercat-protocol](../concepts/ethercat-protocol.md)
- [ethercat-opensource-libs-wechat](../overview/ethercat-opensource-libs-wechat.md)

## 参考来源

- [sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md](../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md)
- [sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md](../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md)

## 推荐继续阅读

- EtherCAT Technology Group：<https://www.ethercat.org/>
- 文章原文：<https://mp.weixin.qq.com/s/C34QLsVb7Sd-GlvPvQlGKQ>
