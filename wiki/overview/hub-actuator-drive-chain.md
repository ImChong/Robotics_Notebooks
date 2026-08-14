---
type: overview
tags: [hub, actuator-drive-chain, actuator, eda, foc, motor-control, ethercat, sim2real, hardware]
status: complete
updated: 2026-08-14
related:
  - ../queries/actuator-drive-chain-selection-loop.md
  - ../concepts/torque-source-abstraction-gap.md
  - ../concepts/implicit-explicit-actuator-modeling.md
  - ../concepts/ethercat-protocol.md
  - ../concepts/field-oriented-control.md
  - ../concepts/joint-friction-models.md
  - ../concepts/motor-torque-current-curve.md
  - ../concepts/motor-torque-speed-curve.md
  - ../concepts/planetary-roller-screw-humanoid-leg-actuation.md
  - ../entities/kicad.md
  - ../entities/altium-designer.md
  - ../entities/simplefoc.md
  - ../entities/paper-neuralactuator-neural-actuation-modeling.md
  - ../entities/bam-better-actuator-models.md
  - ../methods/joint-actuator-parameter-identification.md
  - ../methods/sim2real-joint-sysid-experiment-design.md
  - ../entities/sage-sim2real-actuator-gap-estimator.md
  - ../methods/actuator-network.md
  - ../queries/ethercat-master-optimization.md
  - ../overview/motor-drive-firmware-bus-protocols.md
sources:
  - ../../sources/sites/kicad-org.md
  - ../../sources/sites/altium-designer-primary-refs.md
  - ../../sources/repos/simplefoc_arduino_foc.md
  - ../../sources/papers/neuralactuator_arxiv_2607_11734.md
  - ../../sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md
summary: "执行器驱动链选型闭环知识链枢纽：把 EDA 电路设计 → 电机驱动固件 FOC → 执行器建模与摩擦辨识 → 实时总线闭环集成 四层驱动链，从分散的电子硬件/驱动固件/执行器建模实体页收拢为一条可导航的选型链，统一各层选什么、数据手册标称参数与实测曲线差在哪、建模保真度 vs 辨识成本如何取舍、总线周期 ≠ 闭环带宽的入口。"
---

# 执行器驱动链选型闭环（知识链汇总）

> **知识链定位**：本页是「EDA 电路设计 → 电机驱动固件 FOC → 执行器建模与摩擦辨识 → 实时总线闭环集成」四层驱动链的统一入口，把近周密集 ingest 的 KiCad / Altium / SimpleFOC / NeuralActuator / BAM / SAGE 等电子硬件与执行器建模页从分散的实体页收拢为一条可导航的选型链。它是「[具身大模型分类学选型闭环](./hub-embodied-foundation-model.md)」（选哪一类策略）与「[具身评测基准选型闭环](./hub-embodied-eval-benchmark.md)」（怎么评测/证明它）的**硬件侧姊妹链**——回答「策略算出的力矩指令能不能被真实驱动链忠实执行」。

## 一句话定义

**执行器驱动链选型闭环** 指把策略（RL/MPC）输出的关节力矩指令落到真机时，按 **EDA 电路设计 → 电机驱动固件 FOC → 执行器建模与摩擦辨识 → 实时总线闭环集成** 逐层分工的硬件执行谱系。各层共享「标称参数 ↔ 真机实测」的对齐底座，但在开源 vs 商用、自研 vs 一体化关节、建模保真度 vs 辨识成本、总线周期 vs 闭环带宽上各有取舍，需按落地目标组合选型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EDA | Electronic Design Automation | 电子设计自动化，画原理图/PCB 的工具链 |
| FOC | Field-Oriented Control | 磁场定向控制，电机驱动固件的电流环核心算法 |
| QDD | Quasi-Direct-Drive | 准直驱执行器，低减速比高反驱透明度 |
| BAM | Better Actuator Models | 基于实测的执行器摩擦/力矩辨识框架 |
| EtherCAT | Ethernet for Control Automation Technology | 实时工业以太网现场总线 |
| gap | Sim-to-Real Actuator Gap | 仿真理想执行器与真机执行器的偏差 |

## 为什么重要

- **补一条贯通的驱动链选型视角**：仓库已有各电子硬件/驱动/执行器建模的实体页，但缺「从画驱动板到实时总线逐层选什么、各层标称参数为何对不上实测」的统一决策入口。
- **暴露驱动链层间取舍矛盾**：理想力矩源假设 vs 摩擦/齿隙实际、数据手册峰值力矩 vs 持续力矩热约束、FOC 电流环带宽 vs 编码器分辨率制约、总线周期快 ≠ 闭环带宽高、执行器网络拟合好 vs 分布外温升漂移——这些矛盾只有并置在一条链上才看得清（详见事实库对应驱动链选型矛盾检测规则）。
- **与选型/评测闭环同向**：策略选型解决「算什么力矩」，评测闭环解决「怎么证明它」，驱动链闭环解决「力矩指令能不能被真实硬件忠实执行」，三者构成从算法到硬件的完整落地链。

## 四层驱动链选型闭环

| 层次 | 选什么 | 代表工具/方案 | 站内入口 |
|------|--------|----------------|----------|
| ① EDA 电路设计 | 开源 vs 商用、驱动板自研 vs 商用一体化关节 | KiCad、Altium Designer | [KiCad](../entities/kicad.md)、[Altium Designer](../entities/altium-designer.md) |
| ② 驱动固件 FOC | 电流环带宽/编码器分辨率/标定 | SimpleFOC、FOC 电流环 | [SimpleFOC](../entities/simplefoc.md)、[磁场定向控制](../concepts/field-oriented-control.md) |
| ③ 执行器建模与摩擦辨识 | 显式摩擦模型 vs 神经执行器网络 | BAM、FloBaRoID、NeuralActuator、执行器网络 | [关节执行器参数辨识](../methods/joint-actuator-parameter-identification.md)、[关节动力学辨识实验设计](../methods/sim2real-joint-sysid-experiment-design.md)、[BAM 摩擦辨识](../entities/bam-better-actuator-models.md)、[NeuralActuator](../entities/paper-neuralactuator-neural-actuation-modeling.md)、[隐式/显式执行器建模](../concepts/implicit-explicit-actuator-modeling.md) |
| ④ 实时总线闭环集成 | 总线周期/抖动与控制带宽的关系 | EtherCAT、主站优化 | [EtherCAT 协议基础](../concepts/ethercat-protocol.md)、[EtherCAT 主站优化](../queries/ethercat-master-optimization.md) |
| 端到端 | 四层如何逐层选型取舍 | 选型决策树 | [驱动链选型闭环 Query](../queries/actuator-drive-chain-selection-loop.md) |

## 驱动链选型的关键取舍

- **理想力矩源 vs 真实执行器**：RL/MPC 策略把执行器当理想力矩源，这一抽象在摩擦/齿隙/带宽/热约束下会破——详见[力矩源抽象 gap](../concepts/torque-source-abstraction-gap.md)。
- **标称参数 vs 实测曲线**：数据手册峰值力矩 ≠ 持续可用力矩（热约束）、标称 Kt/Ke ≠ 实测[力矩-电流曲线](../concepts/motor-torque-current-curve.md)与[力矩-转速曲线](../concepts/motor-torque-speed-curve.md)。
- **建模保真度 vs 辨识成本**：显式[摩擦模型](../concepts/joint-friction-models.md)可解释但难覆盖长尾，神经[执行器网络](../methods/actuator-network.md)拟合好但分布外（温升/磨损）会漂移。
- **总线周期 vs 闭环带宽**：总线周期快 ≠ 闭环带宽高，抖动与相位裕度才是控制带宽的真实约束。
- **高减速比 vs 反驱透明度**：高减速比力矩大，但反驱透明度损失、反射惯量上升，QDD 走另一条取舍路线。

## 与其他知识链的关系

- **[通信协议（Communication）](./hub-communication.md)**：④ 层 EtherCAT/CAN 现场总线与通信知识链共享底层数据链路。
- **[物理保真度（Physics Fidelity）](./hub-physics-fidelity.md)**：③ 层执行器建模是物理保真度四层里最靠近硬件的一层，共享 sim2real gap 归因。
- **[接触力控（Contact Force Control）](./hub-contact-force-control.md)**：力矩指令能否忠实执行直接决定力控闭环的下限。

## 关联页面

- [执行器驱动链选型闭环 Query](../queries/actuator-drive-chain-selection-loop.md)
- [力矩源抽象 gap](../concepts/torque-source-abstraction-gap.md)
- [隐式/显式执行器建模](../concepts/implicit-explicit-actuator-modeling.md)
- [KiCad](../entities/kicad.md) · [Altium Designer](../entities/altium-designer.md)
- [SimpleFOC](../entities/simplefoc.md) · [磁场定向控制](../concepts/field-oriented-control.md)
- [BAM 摩擦辨识](../entities/bam-better-actuator-models.md) · [关节执行器参数辨识](../methods/joint-actuator-parameter-identification.md) · [关节动力学辨识实验设计](../methods/sim2real-joint-sysid-experiment-design.md) · [NeuralActuator](../entities/paper-neuralactuator-neural-actuation-modeling.md) · [执行器网络](../methods/actuator-network.md)
- [SAGE sim2real 执行器 gap 估计](../entities/sage-sim2real-actuator-gap-estimator.md)
- [EtherCAT 协议基础](../concepts/ethercat-protocol.md) · [EtherCAT 主站优化](../queries/ethercat-master-optimization.md)
- [电机力矩-电流曲线](../concepts/motor-torque-current-curve.md) · [电机力矩-转速曲线](../concepts/motor-torque-speed-curve.md) · [行星滚柱丝杠腿部执行](../concepts/planetary-roller-screw-humanoid-leg-actuation.md)
- [电机驱动固件与总线协议总览](./motor-drive-firmware-bus-protocols.md)

## 参考来源

- [KiCad 官网](../../sources/sites/kicad-org.md) — 开源 EDA 工具链
- [Altium Designer 一手资料](../../sources/sites/altium-designer-primary-refs.md) — 商用 EDA
- [SimpleFOC 仓库](../../sources/repos/simplefoc_arduino_foc.md) — 开源 FOC 驱动固件
- [NeuralActuator 论文](../../sources/papers/neuralactuator_arxiv_2607_11734.md) — 神经执行器建模
- [BAM-extended 论文](../../sources/papers/bam_extended_friction_servos_arxiv_2410_08650.md) — 执行器摩擦辨识
- 本页归纳自 [驱动链选型闭环 Query](../queries/actuator-drive-chain-selection-loop.md) 及各驱动链实体/概念页
