---
type: entity
tags: [hardware, actuator, motor-control, foc, can, open-source, motionlayer]
status: complete
updated: 2026-07-25
related:
  - ./moteus.md
  - ./simplefoc.md
  - ../comparisons/open-source-qdd-actuator-projects.md
  - ../queries/actuator-drive-chain-selection-loop.md
  - ../concepts/field-oriented-control.md
  - ../../roadmap/depth-torque-motor-design.md
sources:
  - ../../sources/repos/tinymovr.md
  - ../../sources/personal/open_source_qdd_actuator_learning_curator.md
summary: "Tinymovr（现 motionlayer）：紧凑开源无刷驱动，含原理图、PCB、固件、FOC、绝对编码器、CAN 与 Python 上位机，适合从零学习小型关节驱动器。"
---

# Tinymovr（紧凑开源关节驱动）

## 一句话定义

**Tinymovr** 是紧凑型开源无刷电机控制器（历史组织 `tinymovr`，现主仓 [motionlayer/Tinymovr](https://github.com/motionlayer/Tinymovr)）：公开原理图、PCB、固件、FOC、绝对编码器、CAN 与 Python 上位机工具。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FOC | Field-Oriented Control | 无刷电机的磁场定向控制 |
| CAN | Controller Area Network | 电机/关节常用的现场总线通信协议 |
| PCB | Printed Circuit Board | 印制电路板 |
| BLDC | Brushless DC Motor | 无刷直流电机 |
| API | Application Programming Interface | 上位机/脚本调用接口 |

## 为什么重要

- 相对 [moteus](./moteus.md)，文档与板级体量更偏「小型驱动从零学起」。
- 在开源执行器学习阶梯中，与 moteus 同属 **Stage 2：关节驱动 PCB 与固件**。

## 核心结构/机制

| 资产 | 说明 |
|------|------|
| 硬件 | 原理图 + PCB |
| 固件 | FOC 电流/运动控制 |
| 传感 | 绝对编码器集成叙事 |
| 总线 | CAN |
| 工具 | Python 上位机 |

## 工程实践

- 对照阅读：功率回路、电流采样、编码器对齐、CAN 帧——再决定是否自研改板。
- 许可为 **GPL-3.0**：商用闭源固件分发前需评估传染性。

## 局限与风险

- 峰值电流与热设计面向紧凑应用，不宜默认外推到重型人形髋膝峰值。
- 组织迁至 motionlayer 后，以当前 README/发行版为准核对 API 变更。

## 关联页面

- [moteus](./moteus.md)
- [SimpleFOC](./simplefoc.md)
- [开源 QDD 执行器项目对比](../comparisons/open-source-qdd-actuator-projects.md)

## 参考来源

- [sources/repos/tinymovr.md](../../sources/repos/tinymovr.md)
- [开源 QDD 执行器学习策展](../../sources/personal/open_source_qdd_actuator_learning_curator.md)

## 推荐继续阅读

- <https://github.com/motionlayer/Tinymovr>
- <https://motionlayer.company>
