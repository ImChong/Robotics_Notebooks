---
type: entity
tags: [hardware, motor, axial-flux, pcb, open-source, kicad, pmsm, bldc]
status: complete
updated: 2026-07-25
related:
  - ../comparisons/open-source-torque-motor-em-design.md
  - ./axfluxmdo.md
  - ./kicad.md
  - ./ironless-qdd-actuator.md
  - ./cadenkraft-ironless-axial-flux-motor.md
  - ../concepts/halbach-array.md

  - ./pyleecan.md
  - ../../roadmap/depth-torque-motor-design.md
sources:
  - ../../sources/repos/pcb_motor.md
  - ../../sources/personal/open_source_torque_motor_em_design_curator.md
  - ../../sources/blogs/cadenkraft_coreless_axial_flux_motor_part1.md
summary: "pcb-motor：开源 PCB 定子轴向磁通 PMSM/BLDC（WIP，MIT）；公开槽极/绕组拓扑与 KiCad；约 20 极、6 层、铜 140 μm、气隙 1 mm；适合微型关节学习，不适合人形髋膝。"
---

# PCB Motor（PCB 定子轴向磁通电机）

## 一句话定义

**PCB Motor**（[ziteh/pcb-motor](https://github.com/ziteh/pcb-motor)）是 **PCB 定子** 的轴向磁通 PMSM/BLDC 开源硬件项目（标记 **WIP**）：用多层铜箔绕组代替传统漆包线槽绕组，并公开 KiCad 与绕组拓扑分析要点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PCB | Printed Circuit Board | 印刷电路板；此处作定子绕组载体 |
| AF-PMSM | Axial-Flux Permanent Magnet Synchronous Machine | 轴向磁通永磁同步电机 |
| THD | Total Harmonic Distortion | 总谐波畸变；反电势/转矩谐波指标 |
| Halbach | Halbach Array | 单侧聚磁永磁阵列 |
| KiCad | KiCad EDA | 开源原理图/PCB 设计套件 |

## 为什么重要

- 展示 **PCB 绕组如何替代漆包线**：层数、铜厚、线宽与相电阻、气隙与力矩的工程权衡可读。
- 把轴向磁通从「概念」落到 **可打样 Gerber**（见 [KiCad](./kicad.md)），适合手指、腕、灵巧手、云台、微型执行器。
- 与 [axfluxmdo](./axfluxmdo.md) 工具链互补：一边是可制造样例，一边是连续优化空间。
- 与 [Caden Axial Flux Part 1](./cadenkraft-ironless-axial-flux-motor.md)（漆包线线圈 + 打印结构、**未开源 CAD**）对照：本仓提供 **可打样 Gerber**，彼仓提供 Halbach 手算叙事。

## 核心信息（README 摘录）

| 项 | 内容 |
|----|------|
| 状态 | **WIP** |
| 极数 | 约 **20** |
| PCB | **6** 层，厚 **2 mm**，铜 **140 μm** |
| 气隙 | **1 mm** |
| 转子 | 可考虑 Halbach |
| 资产 | `pcb-stator-radial/` KiCad 工程 |
| 许可 | MIT |

绕组拓扑对照（文献综述写入 README）：同心 / 平行 / 径向 / 弧形 / 不等宽平行——在电阻、转矩、反电势与谐波上各有取舍；不等宽平行可降相电阻。

## 核心原理

```mermaid
flowchart LR
  pcb["多层 PCB 定子\n铜箔绕组"]
  gap["气隙 ~1 mm"]
  rot["永磁转子\n可选 Halbach"]
  pcb --> gap --> rot
```

轴向磁通主磁路穿过气隙沿轴向闭合；力矩半径由 PCB 外径决定，适合扁薄包络，但铜截面与散热路径不同于槽满率漆包线电机。

## 工程实践

| 学习点 | 做法 |
|--------|------|
| 打开工程 | 用 KiCad 打开 `pcb-stator-radial.kicad_pcb`，看层叠与绕组走线 |
| 对比拓扑 | 对照 README 引用论文表，理解为何选径向/不等宽 |
| 应用边界 | 只用于小力矩关节原型；髋膝仍走径向外转子 + 减速或更大电机 |

## 局限与风险

- **WIP**：成熟样机测试与完整热设计不足。
- **力矩量级**不支撑人形髋/膝；铜箔损耗与温升在大电流下更苛刻。
- FEM 文件完整度低于 Ironless/FEMM-FOC；电磁验证需自建或配合 axfluxmdo/商业 FEA。

## 关联页面

- [开源力矩电机电磁设计完整度对比](../comparisons/open-source-torque-motor-em-design.md)
- [axfluxmdo](./axfluxmdo.md) · [KiCad](./kicad.md) · [Ironless QDD](./ironless-qdd-actuator.md)
- [Caden Kraft Ironless Axial Flux Motor](./cadenkraft-ironless-axial-flux-motor.md)
- [Halbach Array](../concepts/halbach-array.md)


## 参考来源

- [sources/repos/pcb_motor.md](../../sources/repos/pcb_motor.md)
- [开源力矩电机电磁设计策展](../../sources/personal/open_source_torque_motor_em_design_curator.md)
- [Caden Kraft Axial Flux Part 1](../../sources/blogs/cadenkraft_coreless_axial_flux_motor_part1.md)

## 推荐继续阅读

- 仓库：<https://github.com/ziteh/pcb-motor>
- 灵感项目：sabanekko3/pcb_stator_v2、CarlBugeja/PCB-Motor-v4（README 引用）
