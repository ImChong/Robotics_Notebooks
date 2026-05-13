---
type: entity
tags: [humanoid, hardware, cad, pcb, bom, roboparty]
status: complete
updated: 2026-04-25
related:
  - ./roboto-origin.md
  - ./open-source-humanoid-hardware.md
  - ./humanoid-robot.md
  - ../concepts/text-to-cad.md
sources:
  - ../../sources/repos/atom01_hardware.md
summary: "Atom01_hardware 是 Roboparty Atom01 的机械与电子设计仓库，包含 CAD/PCB/BOM 等复现机器人本体的关键资料。"
---

# Atom01 Hardware

**Atom01_hardware** 是 Roboparty Atom01 机器人的硬件主仓库，负责承载机械结构、电子设计与物料清单等“实体可复现”资产。

## 为什么重要

- 它定义了 Atom01 实体平台的“物理边界条件”（尺寸、质量、关节布局、执行器装配）。
- 是 `atom01_description`（URDF）与 `atom01_firmware`（底层执行）的上游基础。
- 对 DIY 人形平台的成本评估、维护难度评估、可替代器件选型有直接意义。

## 核心结构/机制

- **结构设计层**：CAD 与装配结构文件。
- **电子设计层**：PCB 与接口布局。
- **供应链层**：BOM 与采购映射。

## 常见误区或局限

- 误区：有了硬件仓库就能直接跑策略。实际上还需要 `atom01_deploy`、`atom01_firmware` 与训练模型闭环。
- 局限：开源硬件文档通常偏原型阶段，工业级可靠性验证仍需大量自测。

## 参考来源

- [sources/repos/atom01_hardware.md](../../sources/repos/atom01_hardware.md)
- [Roboparty/Atom01_hardware](https://github.com/Roboparty/Atom01_hardware)

## 关联页面

- [Roboto Origin（开源人形机器人基线）](./roboto-origin.md)
- [开源人形机器人硬件方案对比](./open-source-humanoid-hardware.md)
- [Humanoid Robot](./humanoid-robot.md)
- [文字生成 CAD（Text-to-CAD）](../concepts/text-to-cad.md)

## 推荐继续阅读

- [Atom01 Description](./atom01-description.md)
- [Atom01 Firmware](./atom01-firmware.md)
