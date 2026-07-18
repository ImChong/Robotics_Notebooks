# KiCad（kicad.org）

> 来源归档

- **标题：** KiCad — Schematic Capture & PCB Design Software
- **类型：** site（开源 EDA 套件官网）
- **来源：** KiCad Services Corporation / KiCad 社区
- **链接：** https://www.kicad.org/
- **文档：** https://docs.kicad.org/（含 [10.0 简体中文](https://docs.kicad.org/10.0/zh/)）
- **论坛：** https://forum.kicad.info/
- **入库日期：** 2026-07-18
- **代码：** 主开发仓库 [GitLab `kicad/code/kicad`](https://gitlab.com/kicad/code/kicad)（已开源）；GitHub [KiCad/kicad-source-mirror](https://github.com/KiCad/kicad-source-mirror) 为只读镜像，**不接受 PR**
- **一句话说明：** 跨平台 **GPLv3 开源 EDA**：原理图捕获、PCB 布局、集成 SPICE 与 ERC、Gerber 检视、3D 预览与 **kicad-cli** 命令行；机器人栈中用于关节驱动板、BMS、主控/传感转接板等 **硬件电子设计上游**。
- **沉淀到 wiki：** [KiCad](../../wiki/entities/kicad.md)、[力矩电机设计纵深 Stage 4](../../roadmap/depth-torque-motor-design.md)、[Humanoid Hardware 101 · 05 PCB](../../wiki/overview/humanoid-hardware-101-power-compute-electronics.md)

---

## 平台定位（官网摘要）

KiCad 是面向 **从原理图到可制造 PCB** 的完整桌面套件，强调：

| 模块 | 能力 |
|------|------|
| **原理图编辑器（Schematic Editor）** | 自底向上到 **数百页层次化原理图**；官方符号库 + 自定义符号；**集成 SPICE 仿真** 与 **电气规则检查（ERC）** |
| **PCB 编辑器（PCB Editor）** | 交互式布线器、改进的选择/可视化；现代高密度板布局 |
| **3D 查看器** | 机械配合检查、成品预览；内置光线追踪渲染 |
| **Gerber 查看器** | 制造前检视 CAM 输出 |
| **PCB 计算器** | 走线载流、过孔、阻抗等工程计算（含 IPC-2152 等公式更新） |
| **kicad-cli** | 无头导出 Gerber、DRC、批处理（CI 友好） |

**不是** 机械 CAD（见 [FreeCAD](../../wiki/entities/freecad.md)）、**不是** MCU 固件仿真器（见 [Wokwi](../../wiki/entities/wokwi.md)），而是 **电子设计真值** 与 **Gerber/BOM 制造交付** 层。

---

## 版本与社区（2026-07 官网）

- 稳定线：**10.0.x**（如 10.0.4、10.0.5-rc1 等博客发布）
- 年度社区大会 **KiCon**（如 KiCon Europe 2026）
- 符号/封装库由社区与官方库项目维护（与主程序分仓）

---

## 与机器人栈的关系

| 场景 | 价值 |
|------|------|
| **关节/电机驱动 PCB** | 三相桥、栅极驱动、相电流采样、CAN/EtherCAT 收发器落板；对齐 [力矩电机设计纵深 Stage 4](../../roadmap/depth-torque-motor-design.md) |
| **电源与 BMS** | 电池包监测、DC-DC、保护电路；见 [Humanoid Hardware 101 · 05](../../wiki/overview/humanoid-hardware-101-power-compute-electronics.md) |
| **主控/传感转接** | IMU、编码器、调试 UART/SWD 接口板；与 [电机底软通信总览](../../wiki/overview/motor-drive-firmware-bus-protocols.md) 的 L1 调试层衔接 |
| **开源整机复现** | [EN02-OP](../../wiki/entities/en02-op.md)、[Asimov V1](../../wiki/entities/asimov-v1.md) 等仓库常附 Gerber/原理图；KiCad 是阅读/改版此类资产的通用工具 |
| **制造交付** | 导出 Gerber + 钻孔 + BOM → JLCPCB/嘉立创等打样；DFM 与 [开源人形硬件](../../wiki/entities/open-source-humanoid-hardware.md) 降本叙事一致 |

---

## 外部权威引用

- [KiCad 官网](https://www.kicad.org/)
- [KiCad 文档 10.0 简体中文](https://docs.kicad.org/10.0/zh/)
- [KiCad 开发者文档](https://dev-docs.kicad.org/)
- [KiCad Forum](https://forum.kicad.info/)
