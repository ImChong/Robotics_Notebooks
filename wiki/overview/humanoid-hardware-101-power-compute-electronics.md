---
type: overview
tags: [humanoid, hardware, battery, compute, pcb, bms, category-hub]
status: complete
updated: 2026-11-16
summary: "人形机器人能源与计算电子：按任务占空比做电池、算力、配电与热预算；以 TI 电池监测、NVIDIA Jetson Thor 与 KiCad 官方资料区分器件规格、系统能力和制造交付。"
related:
  - ./humanoid-hardware-101-technology-map.md
  - ../queries/humanoid-battery-thermal-management.md
  - ../entities/open-source-humanoid-brains.md
  - ../entities/kicad.md
  - ../entities/altium-designer.md
original:
  - ../../sources/sites/robot-energy-compute-pcb-review.md
  - ../../sources/courses/kicad_docs_10_zh.md
sources:
  - ../../sources/blogs/wechat_human_five_humanoid_hardware_101.md
  - ../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md
---

# Humanoid Hardware 101 · 05：能源与计算电子

> **分类页**：**电池、计算单元、PCB** 三章 — 移动人形的 **能量预算** 与 **边缘算力**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| NMC | Nickel Manganese Cobalt | 三元锂电池正极材料体系 |
| PCB | Printed Circuit Board | 印刷电路板 |
| BMS | Battery Management System | 电池管理系统 |
| DFM | Design for Manufacturing | 面向制造的设计，降低量产成本与风险 |

## 电池

- 电芯化学体系、串数、倍率和温度范围按任务选择；原综述的 NMC 行业判断不作为所有人形机器人的固定配置。
- 人形需同时：**高功率密度**（失稳毫秒级修正）+ **高能量密度**（整班续航）——化学体系 **难以双极致**。
- 电池包增重 → 更大电机扭矩 → 更高功耗（**重量螺旋**）；优化在 **标准电芯 + 封装集成 + 认证 BMS**。

## 计算单元

- 以 **NVIDIA Jetson Thor T5000** 为一个官方可核对的案例：128 GB 内存、273 GB/s 带宽、40–130 W 功率范围，最高 2070 **FP4 sparse TFLOPS**；这不代表任意模型的实测速率，也不证明市场占有率。
- 早期定制板卡贵，**模块复用 + DFM** 可快速降本（非硅价 alone）。

## PCB

- 驱动板、控制板、通信与电源管理；受益于 **面板化批量** 与汽车/移动电子生态。
- 自研或改版开源板卡时，常用 **[KiCad](../entities/kicad.md)** 等开源 EDA 完成原理图 → layout → Gerber 交付；量产向流程可参考商业 EDA 官方文档 **[Altium Designer](../entities/altium-designer.md)**（规则驱动 layout、制造 OutJob、ECAD-MCAD 协同）。关节驱动落板细节见 [力矩电机设计纵深 Stage 4](../../roadmap/depth-torque-motor-design.md)；低功率开源对照见 [SimpleFOC](../entities/simplefoc.md) 板卡资料。

## 官方资料核对：器件、模块与系统分别验收

[TI 与 NVIDIA 官方资料](../../sources/sites/robot-energy-compute-pcb-review.md)将本页的硬件例子落实到具体边界：BQ76952 面向 3–16 串电池，提供监测、保护及均衡能力，支持多种电池体系；它本身不是完整电池包或认证 BMS，仍需外部开关、采样网络、阈值配置及电芯验证。

Jetson 模块的内存、带宽、功率模式和推理数值应连同精度/稀疏性口径一起记录；计算模块功耗还要叠加载板、存储、传感器和 DC/DC 损耗。不能把 FP4 sparse TFLOPS 与 INT8 TOPS 直接比较，也不能由峰值算力推定持续工作频率或整机续航。

PCB 交付以[已通读的 KiCad 10.0 官方入门](../../sources/courses/kicad_docs_10_zh.md)为依据：保留工程、规则、原理图和布局的一致性，填铜与 DRC 完成后导出 Gerber/钻孔/BOM；工具过检不等于电流、热与 EMC 已通过实板测试。

## 关联页面

- [Humanoid Hardware 101 技术地图](./humanoid-hardware-101-technology-map.md)
- [机器人整机配电架构](../concepts/robot-power-distribution-architecture.md) — 母线/DC-DC/线束/安全回路的整机侧设计
- [人形电池热管理 Query](../queries/humanoid-battery-thermal-management.md)
- [开源人形「大脑」对比](../entities/open-source-humanoid-brains.md)
- [KiCad（开源 PCB EDA）](../entities/kicad.md)
- [Altium Designer（商业 PCB EDA）](../entities/altium-designer.md)
- [产业与成本地缘](./humanoid-hardware-101-supply-chain-economics.md)

## 参考来源

- [TI 电池监测与 NVIDIA Jetson Thor 官方资料复核](../../sources/sites/robot-energy-compute-pcb-review.md) — 本次实际读完的一手资料与核对边界。

- [wechat_human_five_humanoid_hardware_101.md](../../sources/blogs/wechat_human_five_humanoid_hardware_101.md)
- [wechat_humanoid_hardware_101_2026-06-01.md](../../sources/raw/wechat_humanoid_hardware_101_2026-06-01.md)

## 一句话定义

能源与计算电子把电池的能量预算、计算平台的功耗与板卡的配电、保护和散热连成整机的电气底座。

## 为什么重要

电池、计算模块与关节负载共享有限的质量和热预算；板卡设计既要承接峰值功率，又要保证控制与感知持续供电。

## 核心原理

从任务占空比估算平均能耗，以峰值负载校核母线和保护，再分域供电给计算、传感与执行器。PCB 将电源变换、信号接口与测试入口落实为可制造电路。

## 工程实践

建立电源树与各路电压、电流和上电顺序表；带真实计算任务与执行器负载测母线跌落、温升和复位事件。PCB 改版应关联 BOM、固件和测试记录。

## 局限与风险

原文的产品与成本判断带有时间和样机边界；不能把某计算平台或电芯体系当作所有人形的固定配置。续航必须给出运动、待机与作业的比例。

## 推荐继续阅读

先读[整机配电架构](../concepts/robot-power-distribution-architecture.md)，再用[KiCad](../entities/kicad.md)落实板级电路与制造交付。
