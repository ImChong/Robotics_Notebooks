---
type: entity
project_id: soes
code: "https://github.com/OpenEtherCATsociety/SOES"
tags: [ethercat, fieldbus, open-source, slave, embedded, realtime]
status: draft
updated: 2026-11-18
related:
  - ./soem.md
  - ./beckhoff-ssc.md
  - ../concepts/ethercat-protocol.md
  - ../overview/ethercat-opensource-libs-wechat.md
sources:
  - ../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md
  - ../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md
summary: "SOES (Simple Open EtherCAT Slave)：与 SOEM 同组织的轻量开源 EtherCAT 从站协议栈，用于自制从站（传感器、IO 板）。"
---

# SOES (Simple Open EtherCAT Slave)

## 一句话定义

与 SOEM 同组织的轻量开源 EtherCAT 从站协议栈，用于自制从站（传感器、IO 板）。（分类：EtherCAT 从站协议栈；语言/形态：C）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ESC | EtherCAT Slave Controller | 从站硬件芯片（如 ET1100 / LAN9252 / ET1815） |
| EtherCAT | Ethernet for Control Automation Technology | 工业以太网现场总线 |
| CoE | CANopen over EtherCAT | 最常用的 EtherCAT 应用层协议 |

## 为什么重要

- 在 [EtherCAT 开源库清单](../overview/ethercat-opensource-libs-wechat.md) 中属于「从站协议栈」一栏，是机器人关节总线工具链的一环。
- 自研关节/传感器模块要接入 EtherCAT 总线时的开源从站起点；与商用 [Beckhoff SSC](./beckhoff-ssc.md) 构成「开源 vs 官方」选型。

## 核心原理

文章对该项目的描述（以下均为文章声明，未独立核查）：

- 语言：C；文章称是配套 SOEM 的同组织项目。
- 适配 ESC 芯片：ET1100、LAN9252、ET1815。
- 文章称 STM32 + LAN9252 组合非常常用，用于做自定义从站（传感器、IO 板）。
- 文章选型句：自己做硬件从站（薄膜传感器 IO 板）→ SOES + LAN9252。

### 架构流程图

```mermaid
flowchart TD
  BUS["EtherCAT 主站总线\n（如 SOEM）"] --> ESC["ESC 从站硬件芯片\nET1100 / LAN9252 / ET1815"]
  ESC --> HAL["地址偏移 HAL（drivers/）\nESC 寄存器与内存读写接口"]
  HAL --> STACK["SOES 从站核心栈（soes/）\nMailbox · CoE（对象字典 / SDO）· PDO 映射 · FoE / EoE"]
  STACK --> SYNC["同步与调度机制\nSM 同步 · DC sync0 · 轮询 / 中断模式"]
  SYNC --> APP["自定义从站应用（applications/）\n传感器 · IO 板固件"]
```

## 工程实践

- 适用场景：自研关节/传感器模块要接入 EtherCAT 总线时的开源从站起点；与商用 [Beckhoff SSC](./beckhoff-ssc.md) 构成「开源 vs 官方」选型。
- 官方地址：<https://github.com/OpenEtherCATsociety/SOES>（文章给出，本次未独立打开核查）
- 协议背景见 [EtherCAT 协议基础](../concepts/ethercat-protocol.md)；主站实时调优见 [主站优化指南](../queries/ethercat-master-optimization.md)。

## 源码运行时序图

以下时序图基于仓库 `applications/linux_lan9252demo/main.c`、`applications/rtl_slavedemo/main.c` 与 `soes/ecat_slv.c` 源码整理，反映从站协议栈初始化与自由运行轮询周期的实际调用链。

```mermaid
sequenceDiagram
  participant App as 应用入口 (linux_lan9252demo / rtl_slavedemo)
  participant Slv as soes/ecat_slv.c
  participant ESC as ESC HAL (ESC_init / ESC_read / ESC_write)
  participant CoE as CoE 与邮箱模块 (esc_coe)
  participant Hooks as 应用回调 (cb_get_inputs / cb_set_outputs)

  App->>Slv: ecat_slv_init(&config) (rtl_slavedemo 通过 task_spawn 启动)
  Slv->>ESC: ESC_config(config) 与 ESC_init(config)
  Slv->>ESC: 循环 ESC_read(ESCREG_DLSTATUS) 等待 DLstatus & 0x0001
  Slv->>ESC: ESC_ALstatus(ESCinit) / ESC_stopmbx / ESC_stopinput / ESC_stopoutput
  Slv->>CoE: COE_initDefaultValues()

  loop while (1) 自由运行周期
    App->>Slv: ecat_slv()
    Slv->>Slv: ecat_slv_poll()
    Slv->>ESC: ESC_read(ESCREG_LOCALTIME) 与 ESC_state() / ESC_sm_act_event()
    opt ESC_mbxprocess() > 0
      Slv->>CoE: ESC_coeprocess() / ESC_foeprocess() / ESC_eoeprocess() / ESC_xoeprocess()
    end

    Slv->>Slv: DIG_process(WD_FLAG | OUTPUTS_FLAG | APP_HOOK_FLAG | INPUTS_FLAG)
    opt App.state & APPSTATE_OUTPUT 且 ALevent & ESCREG_ALEVENT_SM2
      Slv->>ESC: RXPDO_update() -> ESC_read(ESC_SM2_sma, rxpdo)
      Slv->>CoE: COE_pdoUnpack(rxpdo, sm2mappings, SMmap2)
      Slv->>Hooks: cb_set_outputs() 与 application_hook()
    end
    opt App.state > 0
      Slv->>Hooks: cb_get_inputs()
      Slv->>CoE: TXPDO_update() -> COE_pdoPack(txpdo, sm3mappings, SMmap3)
      Slv->>ESC: ESC_write(ESC_SM3_sma, txpdo)
    end
  end
```

## 局限与风险

- **开源状态：** 文章声明为公开仓库；**本次会话无法访问该仓库，开源程度/许可证未独立核查**，待后续按 ingest 步骤 2.5 补查。
- **推测/待核查：** SOEM 页已记载 SOES 为姊妹从站栈，两处一致；具体许可证与例程需读仓库（待核查）。
- EtherCAT 协议本身不开源，协议栈代码开源，ESC 硬件芯片需单独购买（文章避坑提醒）。

## 关联页面

- [MIT 模式模拟 CSP：前馈优先的内环整定与五关验证](../methods/mit-mode-csp-emulation-tuning.md) — 自研关节若以从站传输位置目标，CSP 与阻抗接口适配须在应用层另行实现，不能由协议栈名称推断。
- [Direct Force Feedback Control（直接力反馈控制）](../methods/direct-force-feedback-control.md) — 自定义传感器 IO 可作为力反馈数据通道，仍需补齐力标定、同步和控制律，通信通路不是力控制器。

- [soem](./soem.md)
- [beckhoff-ssc](./beckhoff-ssc.md)
- [ethercat-protocol](../concepts/ethercat-protocol.md)
- [ethercat-opensource-libs-wechat](../overview/ethercat-opensource-libs-wechat.md)

## 参考来源

- [sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md](../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md)
- [sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md](../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md)

## 推荐继续阅读

- EtherCAT Technology Group：<https://www.ethercat.org/>
- 文章原文：<https://mp.weixin.qq.com/s/C34QLsVb7Sd-GlvPvQlGKQ>
