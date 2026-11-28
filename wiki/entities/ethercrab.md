---
type: entity
project_id: ethercrab
code: "https://github.com/ethercrab-rs/ethercrab"
tags: [ethercat, fieldbus, open-source, master, embedded, realtime]
status: draft
updated: 2026-10-14
related:
  - ./soem.md
  - ./igh-ethercat-master.md
  - ../concepts/ethercat-protocol.md
  - ../overview/ethercat-opensource-libs-wechat.md
sources:
  - ../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md
  - ../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md
summary: "EtherCrab：纯 Rust 实现的 EtherCAT 主站，面向 Rust 嵌入式开发。"
---

# EtherCrab

## 一句话定义

纯 Rust 实现的 EtherCAT 主站，面向 Rust 嵌入式开发。（分类：EtherCAT 主站；语言/形态：Rust）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EtherCAT | Ethernet for Control Automation Technology | 工业以太网现场总线 |
| CoE | CANopen over EtherCAT | 最常用的 EtherCAT 应用层协议 |
| RTOS | Real-Time Operating System | FreeRTOS / RT-Thread 等 |

## 为什么重要

- 在 [EtherCAT 开源库清单](../overview/ethercat-opensource-libs-wechat.md) 中属于「主站」一栏，是机器人关节总线工具链的一环。
- 技术栈已是 Rust 的机器人/嵌入式团队；成熟度低于 [SOEM](./soem.md)，上生产前应自行评估。

## 核心原理

文章对该项目的描述（以下均为文章声明，未独立核查）：

- 语言：Rust；文章称纯 Rust 安全实现、无 unsafe 依赖。
- 文章称适合 Rust 嵌入式开发，项目相对年轻。
- 文章选型句：Rust 项目 → Ethercrab。

### 架构流程图

```mermaid
flowchart TD
  APP["Rust 异步应用（std / no_std）\ntokio · smol · Embassy"] --> MD["MainDevice\nPduStorage（tx / rx / pdu_loop）"]
  MD --> INIT["初始化与配置\ninit_single_group（EEPROM SII · CoE SDO）"]
  INIT --> DC["DC 分布式时钟\n拓扑延迟检测 · 启动静态漂移补偿"]
  DC --> OP["into_op（PRE-OP 到 OP）\ngroup.tx_rx 周期交换 io_raw PDI 与 DC 同步"]
  OP --> TXRX["tx_rx_task 传输层\nraw socket / xdp / io_uring / Windows NPF"]
  TXRX --> SUB["EtherCAT SubDevices\nEK1100 · EL3004 · CiA402/DS402 驱动器"]
```

## 工程实践

- 适用场景：技术栈已是 Rust 的机器人/嵌入式团队；成熟度低于 [SOEM](./soem.md)，上生产前应自行评估。
- 官方地址：<https://github.com/ethercrab-rs/ethercrab>（文章给出，本次未独立打开核查）
- 协议背景见 [EtherCAT 协议基础](../concepts/ethercat-protocol.md)；主站实时调优见 [主站优化指南](../queries/ethercat-master-optimization.md)。

## 源码运行时序图

以下时序图基于仓库 `Cargo.toml`、`examples/ek1100.rs`（标准 `smol` 异步初始化与周期 PDI 回路）以及 `examples/dc.rs`（DC 分布式时钟对齐与 `tx_rx_dc` 周期）源码整理。

```mermaid
sequenceDiagram
  participant App as 应用入口 ek1100.rs / dc.rs
  participant Pdu as PduStorage (PDU_STORAGE)
  participant MD as MainDevice
  participant TxRx as ethercrab::std::tx_rx_task
  participant Grp as SubDeviceGroup
  participant Sub as SubDevice (EK1100 / EL3004 / EL4102)

  App->>Pdu: PDU_STORAGE.try_split() 获取 tx, rx, pdu_loop
  App->>MD: MainDevice::new(pdu_loop, Timeouts, MainDeviceConfig)
  App->>TxRx: 启动 tx_rx_task(&interface, tx, rx, TxRxTaskConfig)
  App->>MD: init_single_group(ethercat_now) 扫描并配置从站
  MD-->>Grp: 返回处于 PRE-OP 状态的 group
  App->>Sub: subdevice.sdo_write / sdo_write_array 配置 SM PDO 映射 (0x1c12 / 0x1c13 / 0x1c32)

  opt 分布式时钟路径 (examples/dc.rs)
    App->>Sub: subdevice.set_dc_sync(DcSync::Sync0 / Sync01)
    App->>Grp: group.into_pre_op_pdi(&maindevice)
    loop 直至 max_deviation < 500ns
      App->>Grp: group.tx_rx_sync_system_time(&maindevice)
      App->>Sub: register_read(RegisterAddress::DcSystemTimeDifference)
    end
    App->>Grp: group.configure_dc_sync(&maindevice, DcConfiguration)
    App->>Grp: into_safe_op() 后调用 request_into_op() 并轮询 tx_rx_dc 直到 all_op()
  end

  App->>Grp: group.into_op(&maindevice) 进入 OP (ek1100.rs)
  loop 5ms 周期任务 (tick_interval.next())
    App->>Grp: group.tx_rx(&maindevice) 或 tx_rx_dc(&maindevice)
    Grp->>TxRx: 经 pdu_loop 与 tx / rx 任务收发 EtherCAT 帧
    App->>Sub: io_raw() / outputs_raw_mut() 读写过程数据
  end

  App->>Grp: 退出前依次调用 into_safe_op -> into_pre_op -> into_init(&maindevice)
```

## 局限与风险

- **开源状态：** 文章声明为公开仓库；**本次会话无法访问该仓库，开源程度/许可证未独立核查**，待后续按 ingest 步骤 2.5 补查。
- **推测/待核查：** 「无 unsafe」为文章原话，未独立核查；是否 no_std、是否支持具体 RTOS 需读仓库（待核查）。
- EtherCAT 协议本身不开源，协议栈代码开源，ESC 硬件芯片需单独购买（文章避坑提醒）。

## 关联页面

- [soem](./soem.md)
- [igh-ethercat-master](./igh-ethercat-master.md)
- [ethercat-protocol](../concepts/ethercat-protocol.md)
- [ethercat-opensource-libs-wechat](../overview/ethercat-opensource-libs-wechat.md)
- [Robot Software System Integration · 机器人软件系统集成](../methods/robot-software-integration.md)：在机器人软件系统中承担现场总线通信层接入与实时设备驱动集成。
- [Robot Control System Architecture · 机器人控制系统架构](../methods/robot-control-architecture.md)：为底层关节控制环提供确定性周期通信与执行器数据通道。

## 参考来源

- [sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md](../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md)
- [sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md](../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md)

## 推荐继续阅读

- EtherCAT Technology Group：<https://www.ethercat.org/>
- 文章原文：<https://mp.weixin.qq.com/s/C34QLsVb7Sd-GlvPvQlGKQ>
