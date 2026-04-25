---
type: entity
tags: [humanoid, firmware, can, real-time, control-stack, roboparty]
status: complete
updated: 2026-04-25
related:
  - ./roboto-origin.md
  - ./atom01-deploy.md
  - ./atom01-hardware.md
sources:
  - ../../sources/repos/atom01_firmware.md
summary: "atom01_firmware 是 Atom01 的底层固件仓库，覆盖板端构建、通信链路与执行守护，是控制栈最底层支撑。"
---

# Atom01 Firmware

**atom01_firmware** 是 Atom01 的底层固件与板端运行仓库，负责通信、设备驱动与基础运行时能力。

## 为什么重要

- 固件层决定控制指令能否稳定、低延迟地落到执行器。
- 是部署层（ROS2）与硬件层（电机/控制板）之间的关键桥梁。
- 直接影响安全性与故障恢复能力。

## 核心结构/机制

- **通信链路**：例如 USB2CAN 等总线通信。
- **板端构建**：嵌入式环境编译与运行部署。
- **守护逻辑**：基础监控、重连与容错机制。

## 常见误区或局限

- 误区：只要高层策略好，底层固件影响不大。实际中固件稳定性往往是系统可用性的硬约束。
- 局限：固件细节强依赖硬件版本，跨平台复用成本可能较高。

## 参考来源

- [sources/repos/atom01_firmware.md](../../sources/repos/atom01_firmware.md)
- [Roboparty/atom01_firmware](https://github.com/Roboparty/atom01_firmware)

## 关联页面

- [Roboto Origin（开源人形机器人基线）](./roboto-origin.md)
- [Atom01 Deploy](./atom01-deploy.md)
- [Atom01 Hardware](./atom01-hardware.md)

## 推荐继续阅读

- [Contact Dynamics](../concepts/contact-dynamics.md)
- [State Estimation](../concepts/state-estimation.md)
