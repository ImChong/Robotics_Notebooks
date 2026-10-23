---
type: entity
project_id: tscircuit
institutions: [tscircuit]
project: https://tscircuit.com/
code: https://github.com/tscircuit/tscircuit
tags: [software, electronics, eda, pcb, hardware, open-source, typescript]
status: complete
updated: 2026-10-23
related:
  - ../overview/hub-actuator-drive-chain.md
  - ./kicad.md
  - ./freecad.md
  - ../overview/humanoid-hardware-101-power-compute-electronics.md
  - ../overview/motor-drive-firmware-bus-protocols.md
sources:
  - ../../sources/sites/tscircuit-com.md
  - ../../sources/repos/tscircuit.md
summary: "tscircuit 是 MIT 开源的 TypeScript/React 风格电子设计工具链，将电路描述为可复用代码并生成原理图、PCB、3D 预览与制造文件；适合机器人驱动板和传感器板的快速原型及版本化协作。"
---

# tscircuit（代码化电子设计工具链）

**tscircuit** 是以 TypeScript/React 组件描述电路的开源电子设计工具链：将芯片、电阻、网络与 PCB 走线等表示为代码，再预览原理图、PCB 和 3D 模型，并导出制造文件。在机器人栈中，它位于 [KiCad](./kicad.md) 所处的电子设计层，适合用代码复用和版本控制迭代关节驱动、电源、MCU 与传感器接口板。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EDA | Electronic Design Automation | 电子设计自动化；原理图与 PCB 工具类别 |
| PCB | Printed Circuit Board | 印刷电路板；承载机器人电源、驱动与接口电路 |
| BOM | Bill of Materials | 物料清单；制造与采购所需的元件列表 |
| CLI | Command-Line Interface | 命令行界面；`tsci` 用于本地预览、检查与导出 |
| DRC | Design Rule Check | 设计规则检查；用于发现制造约束与几何规则问题 |

## 为什么对机器人硬件有用

- **控制板可直接代码化：** 官方示例包含 RP2040 电机控制器、步进电机控制器和 BLDC 控制板；这些板卡与机器人常见的 MCU、驱动、电源和传感器硬件相近。
- **参数化和复用：** TypeScript/React 组件可将重复电路封装为模块，便于对不同电机、电源和连接器配置做版本化迭代。
- **设计评审可进入软件工作流：** 电路源文件可由 Git 管理；本地 CLI 可预览/检查，制造文件可以作为可审查的产物。
- **与既有 EDA 协作：** 官方文档列出 KiCad 等格式导出；可把 tscircuit 作为代码化入口，并在 [KiCad](./kicad.md) 中继续检查或编辑。

## 核心工作流

1. 使用 TypeScript/React 风格的元素定义电路板、器件、网络与连接。
2. 通过核心工具链解析为 Circuit JSON，并生成原理图、PCB 布局和 3D 预览。
3. 用 `tsci dev` 在本地运行预览；通过 CLI 检查或导出设计。
4. 审核制造输出，包括 Gerber、BOM、钻孔与 Pick-and-Place 文件，再送往 PCB 制造/装配。

## 工程实践

- 用组件或参数统一管理相同电源、MCU、连接器和驱动模块，减少硬件变体之间的手动重复。
- 将源代码与导出的制造文件关联提交，明确记录板卡版本、依赖版本及制造参数。
- 对电机驱动 PCB，重点复核栅极驱动、电流采样、功率回路、热设计、隔离与 CAN 等接口；代码生成和预览不能替代专业硬件审查。

## 局限与风险

- **自动布局仍在 beta：** 官方文档明确其算法在持续改进；复杂板卡需要自行设定器件位置、检查布线并反复审阅。
- **可视化不等于电气正确：** 3D 与 PCB 视图不能替代电气规则、信号完整性、EMI/热设计检查或上电验证。
- **不是机器人专用工具：** 项目是通用 EDA 工作流；机械结构仍由 [FreeCAD](./freecad.md) 等机械 CAD 工具处理，固件和实时控制需要独立工具链。
- **制造细节仍需核验：** Gerber、BOM、钻孔与装配数据在投产前应由制造商规则和实物测试复查。

## 关联页面

- [KiCad（开源 PCB 与原理图 EDA）](./kicad.md) — 桌面端电子设计套件；tscircuit 支持与 KiCad 格式互通。
- [FreeCAD（开源参数化机械 CAD）](./freecad.md) — 机械结构设计与板卡外壳/安装件配合。
- [Humanoid Hardware 101 · 能源与计算电子](../overview/humanoid-hardware-101-power-compute-electronics.md) — 机器人驱动、电源与计算电子系统背景。
- [电机驱动器底软通信协议总览](../overview/motor-drive-firmware-bus-protocols.md) — 驱动器 PCB 与固件/CAN 接口协同。

- [执行器驱动链选型闭环](../overview/hub-actuator-drive-chain.md) — 代码化电路设计适用于驱动、电源和传感器板。

## 参考来源

- [tscircuit 官方项目页归档](../../sources/sites/tscircuit-com.md)
- [tscircuit 官方源码仓库归档](../../sources/repos/tscircuit.md)

## 推荐继续阅读

- [官方文档：What is tscircuit?](https://docs.tscircuit.com/)
- [CLI Quickstart](https://docs.tscircuit.com/intro/quickstart-cli)
- [Automatic PCB Layout](https://docs.tscircuit.com/guides/tscircuit-essentials/automatic-pcb-layout)
- [官方源码仓库](https://github.com/tscircuit/tscircuit)
