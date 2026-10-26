---
type: entity
project_id: atopile
project: https://atopile.io/
institutions: [atopile]
code: https://github.com/atopile/atopile
tags: [software, electronics, eda, pcb, hardware, robotics]
status: complete
updated: 2026-10-26
related:
  - ./kicad.md
  - ./heypcb.md
  - ../overview/humanoid-hardware-101-power-compute-electronics.md
  - ../overview/motor-drive-firmware-bus-protocols.md
sources:
  - ../../sources/repos/atopile.md
summary: "atopile 是以声明式代码表达电子设计的语言、编译器与工具链，强调模块复用、约束校验和参数化器件选择，并与 KiCad PCB 布局衔接。"
---

# atopile

[**atopile**](https://github.com/atopile/atopile) 是面向电子硬件的声明式设计语言、编译器与工具链。用户用 `.ato` 文件描述模块、接口、单位、容差和断言，工具链解析约束、选择器件、运行检查，并与 KiCad PCB 设计协作。

## 一句话定义

**代码化电子设计工具链** — 将可复用电路模块和设计约束写入版本管理的代码工程，再编译、校验并连接到 KiCad 布局与制造流程。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EDA | Electronic Design Automation | 原理图与 PCB 设计工具类别 |
| DSL | Domain-Specific Language | 为特定领域设计的语言；atopile 用于电子硬件描述 |
| PCB | Printed Circuit Board | 印刷电路板，承载驱动、电源、传感和计算电路 |
| CLI | Command-Line Interface | 命令行接口；可用 `ato build` 构建项目 |
| BOM | Bill of Materials | 物料清单，属于硬件制造交付内容 |

## 为什么重要

机器人硬件往往要反复调整电机驱动、电源、传感器和计算板。atopile 将电路模块与参数化约束保存在代码工程中，便于复用、版本管理、协作审查，并把电子设计流程接入软件化构建与检查。它补充了 [KiCad](./kicad.md) 的图形化原理图与 PCB 布局流程，适合关注代码化设计和复用的团队评估。

## 核心结构与工作流

1. **描述设计**：在 `.ato` 文件中定义模块、接口、单位、容差和断言。
2. **复用模块与选择器件**：组合设计模块，并使用参数化器件选择能力。
3. **编译与校验**：工具链求解约束、选择元件并运行设计检查。
4. **PCB 布局与制造**：在 KiCad 中继续布局，生成 BOM、制造与装配数据及报告。
5. **协作与迭代**：通过代码工程管理变更；项目支持在本地运行检查，也可将设计检查放入 CI。

README 的快速上手方式是安装 VS Code/Cursor 扩展后打开示例并运行构建，也可按官方安装文档配置 CLI 并执行 `ato build`。KiCad 推荐用于 PCB 布局，但不是开始构建示例的前置条件。

## 工程实践

- 从 [官方示例](https://github.com/atopile/atopile/tree/main/examples) 开始，先确认工具版本、依赖和构建产物，再接入自有板卡。
- 将电路模块、参数和约束与代码一同评审；对接机器人电机驱动板时，重点复核功率回路、电流采样、保护电路、热设计、EMI 和接口引脚。
- 通过官方软件包注册表查找可复用模块；复用时检查模块版本、器件可获得性、许可证及其适用条件。
- 将自动检查视为设计流程的一部分，仍需在 KiCad、器件数据手册和制造商规则下复核关键电气与 PCB 结果。

## 开源状态与局限

atopile 主仓库公开并采用 MIT 许可，包含语言、编译器与工具链代码；官方 README 还链接文档、示例、编辑器扩展和模块注册表。机器人驱动板是否能被某个模块库直接覆盖，需按具体器件与电气约束确认。

atopile 的代码化描述和构建检查不能单独证明电路安全、可制造或已通过实板验证。高电流、高压、热设计、EMI 与安全相关部分仍需硬件工程师检查，并在打样后测试。

## 关联页面

- [KiCad](./kicad.md) — atopile 工具链衔接的 PCB 布局与制造工具
- [HeyPCB（AI PCB 设计）](./heypcb.md) — 自然语言辅助生成电路设计的另一种工具路径
- [Humanoid Hardware 101 · 能源与计算电子](../overview/humanoid-hardware-101-power-compute-electronics.md) — PCB 在机器人整机硬件中的位置
- [电机驱动器底软通信协议总览](../overview/motor-drive-firmware-bus-protocols.md) — 电路设计中的驱动器总线与接口背景

## 参考来源

- [atopile 官方仓库归档](../../sources/repos/atopile.md)

## 推荐继续阅读

- [atopile 官方文档](https://docs.atopile.io/) — 安装、语言、构建与硬件设计指南
- [atopile GitHub 仓库](https://github.com/atopile/atopile) — 查看源码、示例与开发说明
- [KiCad 官方文档](https://docs.kicad.org/) — 了解 PCB 布局、检查和制造输出
