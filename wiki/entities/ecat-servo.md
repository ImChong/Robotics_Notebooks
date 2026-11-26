---
type: entity
project_id: ecat-servo
code: "https://github.com/kubabuda/ecat_servo"
tags: [ethercat, fieldbus, open-source, servo, embedded, realtime]
status: draft
updated: 2026-11-18
related:
  - ./soes.md
  - ./opensml.md
  - ../concepts/ethercat-protocol.md
  - ../overview/ethercat-opensource-libs-wechat.md
sources:
  - ../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md
  - ../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md
summary: "ecat_servo：实现 CiA402 伺服状态机的 EtherCAT 伺服配套库。"
---

# ecat_servo

## 一句话定义

实现 CiA402 伺服状态机的 EtherCAT 伺服配套库。（分类：EtherCAT 伺服 CiA402 配套；语言/形态：C（CiA402 伺服状态机））

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CiA402 | CAN in Automation 402 drive profile | 伺服驱动器状态机与运动模式标准 |
| CoE | CANopen over EtherCAT | 最常用的 EtherCAT 应用层协议 |
| ESC | EtherCAT Slave Controller | 从站硬件芯片（如 ET1100 / LAN9252 / ET1815） |

## 为什么重要

- 在 [EtherCAT 开源库清单](../overview/ethercat-opensource-libs-wechat.md) 中属于「伺服 CiA402 配套」一栏，是机器人关节总线工具链的一环。
- 自研 EtherCAT 伺服驱动器时参考 CiA402 状态机的开源实现；搭配 [SOES](./soes.md)。

## 核心原理

文章对该项目的描述（以下均为文章声明，未独立核查）：

- 文章称为 CiA402 伺服状态机实现，归入「伺服 CiA402 运动控制配套库」。
- 仓库地址由文章给出。

### 架构流程图

```mermaid
flowchart TD
  BUS["EtherCAT 总线 / CANbus\nPDO 控制字 controlword 与状态字 Status_Word"] --> ESC["ESC 从站硬件芯片\nLAN9252 / AX58100"]
  ESC --> SOES["SOES 从站协议栈\nCoE 对象字典与 PDO 交换（STM32 F1/F4）"]
  SOES --> SM["cia402device 状态机\ncia402_axis_t · cia402_state_machine"]
  SM --> FLAGS["状态标志输出\naxis_func_enabled · hv_power_applied · brake_applied · FAULT_REACTION_ACTIVE"]
  FLAGS --> DRV["电机控制器应用\nODrive / STMBL 等驱动执行"]
```

## 工程实践

- 适用场景：自研 EtherCAT 伺服驱动器时参考 CiA402 状态机的开源实现；搭配 [SOES](./soes.md)。
- 官方地址：<https://github.com/kubabuda/ecat_servo>（文章给出，本次未独立打开核查）
- 协议背景见 [EtherCAT 协议基础](../concepts/ethercat-protocol.md)；主站实时调优见 [主站优化指南](../queries/ethercat-master-optimization.md)。

## 源码运行时序图

以下时序图基于仓库 `cia402device/src/main.c`、`cia402device/lib/cia402device/cia402device.c` 与 `examples/EasyCAT_SPL/src/main.c` 源码整理，展示 CiA402 状态机驱动流程及 STM32/EasyCAT 固件主循环入口。

```mermaid
sequenceDiagram
  participant EasyCAT as examples/EasyCAT_SPL/src/main.c
  participant TestMain as cia402device/src/main.c
  participant SM as cia402device.c 状态机
  participant Axis as cia402_axis_t / Status_Word

  Note over EasyCAT: STM32 固件入口 SysTick_Config(1ms) -> Delay_init() -> ECAT_AppSetup()
  loop EasyCAT 主循环 while (1)
    EasyCAT->>EasyCAT: ECAT_Application() 周期执行从站应用
  end

  TestMain->>SM: cia402_initialize(&cia402axis, &mockOD_Status_Word, &mockOD_ALstatus)
  SM->>Axis: state = NOT_READY_TO_SWITCH_ON, transition = NO_TRANSITION, 清零 flags
  TestMain->>SM: cia402_state_machine(&cia402axis, controlword)
  SM->>Axis: 重置 *statusword = 0x0000, prevflags = flags

  alt *ALstatus == AL_STATUS_OP 且处于 NOT_READY_TO_SWITCH_ON
    SM->>Axis: 状态迁移 1 -> SWITCH_ON_DISABLED
  else SWITCH_ON_DISABLED 收到 SHUTDOWN 或 AL_STATUS_OP
    SM->>Axis: 状态迁移 2 -> READY_TO_SWITCH_ON
  else READY_TO_SWITCH_ON 收到 SWITCH_ON / SWITCH_ON_ENABLE
    SM->>Axis: 状态迁移 3 (-> SWITCHED_ON) 或 3+4 (-> OPERATION_ENABLED)
  else OPERATION_ENABLED 收到 DISABLE_OPERATION / QUICK_STOP / *ALstatus != AL_STATUS_OP
    SM->>Axis: 分别迁移至 SWITCHED_ON / QUICK_STOP_ACTIVE / SWITCH_ON_DISABLED
  else FAULT_REACTION_ACTIVE 或 FAULT 收到 FAULT_RESET
    SM->>Axis: 自动迁移 14 -> FAULT, 复位迁移 15 -> SWITCH_ON_DISABLED
  end

  SM->>Axis: 按当前 axis->state 更新 config_allowed, axis_func_enabled, hv_power_applied, brake_applied
```

## 局限与风险

- **开源状态：** 文章声明为公开仓库；**本次会话无法访问该仓库，开源程度/许可证未独立核查**，待后续按 ingest 步骤 2.5 补查。
- **推测/待核查：** 从文章描述推测其面向从站侧（做伺服驱动器的 EtherCAT 固件），是否同时提供主站侧封装需读仓库（推测，未核查）。
- EtherCAT 协议本身不开源，协议栈代码开源，ESC 硬件芯片需单独购买（文章避坑提醒）。

## 关联页面

- [PID Control（比例-积分-微分控制）](../methods/pid-control.md) — CiA402 状态机管理伺服使能与模式，电机反馈环仍需应用或驱动器实现。
- [MIT 模式模拟 CSP：前馈优先的内环整定与五关验证](../methods/mit-mode-csp-emulation-tuning.md) — 对照 CSP 位置目标与关节阻抗接口适配，需先核实 PDO、模式和控制周期；本页二手入口不证明完整实现。

- [soes](./soes.md)
- [opensml](./opensml.md)
- [ethercat-protocol](../concepts/ethercat-protocol.md)
- [ethercat-opensource-libs-wechat](../overview/ethercat-opensource-libs-wechat.md)

## 参考来源

- [sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md](../../sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md)
- [sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md](../../sources/raw/wechat_ethercat_opensource_libs_qt_adventure_2026-10-14.md)

## 推荐继续阅读

- EtherCAT Technology Group：<https://www.ethercat.org/>
- 文章原文：<https://mp.weixin.qq.com/s/C34QLsVb7Sd-GlvPvQlGKQ>
