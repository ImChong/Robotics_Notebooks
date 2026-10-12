# CADCraft 项目页（ArtCraft）

> 来源归档（官方产品页）

- **标题：** CADCraft: Computer-aided design and drafting, reimagined in pure Rust
- **类型：** site（官方产品页 / 下载与项目概览）
- **URL：** <https://getartcraft.com/apps/cadcraft>
- **代码：** <https://github.com/storytold/cadcraft>
- **入库 / 核查日期：** 2026-10-12
- **开放状态：** 已开源。产品页提供源码入口与多平台下载；仓库声明 MIT OR Apache-2.0，并包含 Rust workspace、CLI、MCP 与 WASM app。
- **页面版本：** 下载区域显示 0.5.0；该页面内容包含源代码运行命令。
- **一句话说明：** ArtCraft 官方 CADCraft 页面介绍其桌面 / Web CAD 绘图能力、支持的图纸格式、下载方式及面向 AI agent 的命令行与 MCP 使用入口。

## 官方页核查结果

- 产品页将 CADCraft 描述为纯 Rust 的计算机辅助设计与绘图应用，面向使用者与 AI agent。
- 页面列举常用绘图流程、DXF（ASCII / binary，R12–2018）、DWG（R13–2018）读取与 PDF / SVG / PNG 导出、paper space / viewports、尺寸标注、hatch / blocks、parametric constraints。
- 页面称 macOS、Windows、Linux 与 Web 可用，并提供源代码构建方式；README 还列 FreeBSD 支持与 CLI / MCP 接口。
- 下载卡片显示 0.5.0，而仓库主分支 Cargo workspace 元数据为 0.6.2；本次将二者分别标注，可能反映页面发行包与开发分支不同步。

## 对 wiki 的映射

- 主实体：[CADCraft](../../wiki/entities/cadcraft.md)
- 官方代码仓归档：[storytold-cadcraft.md](../repos/storytold-cadcraft.md)
