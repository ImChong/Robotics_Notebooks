---
type: entity
project_id: beckhoff-ssc
tags: [ethercat, fieldbus, open-source, slave, embedded, realtime]
status: draft
updated: 2026-10-14
related:
  - ./soes.md
  - ../concepts/ethercat-protocol.md
  - ../concepts/ethercat-protocol.md
  - ../overview/ethercat-opensource-libs-wechat.md
sources:
  - ../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md
  - ../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md
summary: "Beckhoff SSC (EtherCAT Slave Stack Code)：倍福官方 EtherCAT 从站栈源码，ETG 会员免费获取，不在公开 GitHub 仓库发布。"
---

# Beckhoff SSC (EtherCAT Slave Stack Code)

## 一句话定义

倍福官方 EtherCAT 从站栈源码，ETG 会员免费获取，不在公开 GitHub 仓库发布。（分类：EtherCAT 从站协议栈；语言/形态：C（官方从站栈源码））

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ETG | EtherCAT Technology Group | 协议官方组织 |
| ESC | EtherCAT Slave Controller | 从站硬件芯片（如 ET1100 / LAN9252 / ET1815） |
| EtherCAT | Ethernet for Control Automation Technology | 工业以太网现场总线 |

## 为什么重要

- 在 [EtherCAT 开源库清单](../overview/ethercat-opensource-libs-wechat.md) 中属于「从站协议栈」一栏，是机器人关节总线工具链的一环。
- 要通过 EtherCAT 一致性测试或做量产从站时的主流选择；原型阶段可先用 [SOES](./soes.md)。

## 核心原理

文章对该项目的描述（以下均为文章声明，未独立核查）：

- 文章称为官方从站栈，ETG 会员免费获取源码，工业标准、功能最全。
- 文章称需搭配 ESC 芯片，商用项目首选。
- 文章明确：非 GitHub 公开仓库，需在 ETG 官网申请。

## 工程实践

- 适用场景：要通过 EtherCAT 一致性测试或做量产从站时的主流选择；原型阶段可先用 [SOES](./soes.md)。
- 官方地址：文章未给出，本页不臆造 URL
- 协议背景见 [EtherCAT 协议基础](../concepts/ethercat-protocol.md)；主站实时调优见 [主站优化指南](../queries/ethercat-master-optimization.md)。

## 局限与风险

- **开源状态：** **源码受限分发**：非公开仓库，ETG 会员向官网申请（文章声明）。
- **推测/待核查：** 「倍福」为 Beckhoff 中文名，由我补充；SSC 的具体授权条款以 ETG 协议为准（未核查）。
- **推测/待核查：** 按步骤 2.5 口径应归类为「源码受限分发」而非「已开源」。
- EtherCAT 协议本身不开源，协议栈代码开源，ESC 硬件芯片需单独购买（文章避坑提醒）。

## 关联页面

- [soes](./soes.md)
- [ethercat-protocol](../concepts/ethercat-protocol.md)
- [ethercat-protocol](../concepts/ethercat-protocol.md)
- [ethercat-opensource-libs-wechat](../overview/ethercat-opensource-libs-wechat.md)

## 参考来源

- [sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md](../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md)
- [sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md](../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md)

## 推荐继续阅读

- EtherCAT Technology Group：<https://www.ethercat.org/>
- 文章原文：<https://mp.weixin.qq.com/s/C34QLsVb7Sd-GlvPvQlGKQ>
