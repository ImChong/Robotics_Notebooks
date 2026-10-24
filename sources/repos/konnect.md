---
type: source
source_type: repo
title: "Konnect — AI-assisted PCB design for KiCad"
url: "https://github.com/mixelpixx/Konnect"
project_id: konnect-ai-kicad-mcp
project: "https://github.com/mixelpixx/Konnect"
code: "https://github.com/mixelpixx/Konnect"
license: AGPL-3.0
status: public
updated: 2026-10-24
summary: "面向 KiCad 10 的原生 Rust/MCP 插件，让 AI 助手通过 KiCad IPC API 编辑原理图与 PCB，并运行检查、布线和制造导出流程。"
---

# Konnect

> 来源归档

- **类型：** repo（KiCad 原生 MCP 插件）
- **官方仓库：** https://github.com/mixelpixx/Konnect
- **目标平台：** KiCad 10
- **主要实现：** Rust 单一二进制；MCP JSON-RPC stdio，另支持 Streamable HTTP
- **许可证：** AGPL-3.0；README 提及另有商业许可
- **入库日期：** 2026-10-24
- **一句话说明：** 通过 MCP 将 AI 助手连接到 KiCad，支持原理图编辑、PCB 布局与布线、设计检查、器件检索及制造文件导出。
- **沉淀到 wiki：** [Konnect](../../wiki/entities/konnect.md)、[KiCad](../../wiki/entities/kicad.md)

---

## 项目定位与能力

Konnect 是 KiCad 10 的原生插件及 MCP 服务端。README 将其定位为 KiCAD-MCP-Server 的后继项目：将调用路径收敛到 Rust 单进程，并通过 KiCad IPC API 与 PCB 编辑器交互；原理图工程则通过 S-expression 编辑器处理。MCP 客户端可按需加载工具组，覆盖原理图、PCB 编辑、设计审查、JLCPCB 元件检索、参考电路及 Gerber、BOM、坐标等制造输出。

## 开源状态（2026-10-24 核查）

**已开源。** GitHub 主仓库公开 Rust 源码，仓库及 README 标注 AGPL-3.0；README 提供源代码构建说明与 KiCad Plugin and Content Manager 安装包。README 将当前版本标记为 **Beta**，并说明 KiCad 10 为目标版本。使用或分发前应阅读许可证；项目方另提供商业许可选项。

## 关键实现与边界

- PCB 编辑通过 KiCad IPC API 与运行中的 PCB Editor 协作，README 描述其支持实时编辑和 KiCad 撤销/重做。
- 原理图编辑面向 `.kicad_sch` 文件，包含原子写入、UUID 保留与往返测试。
- `kicad-cli` 用于 ERC、DRC、Gerber、PDF 等检查或导出；部分 PCB 操作需要打开目标板卡。
- 项目处于 Beta 阶段；工具生成或修改的板卡仍需工程师核对原理图、器件引脚、供电、间距、热设计、EMI 与制造规则，再进行实板验证。

## 一手来源

- [Konnect 官方 GitHub 仓库与 README](https://github.com/mixelpixx/Konnect)
- [Konnect Releases](https://github.com/mixelpixx/Konnect/releases)
- [Model Context Protocol](https://modelcontextprotocol.io/)
