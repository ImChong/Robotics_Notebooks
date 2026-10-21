# opengeos / RustGIS

- **标题：** RustGIS
- **类型：** repo
- **来源：** opengeos
- **链接：** <https://github.com/opengeos/RustGIS>
- **项目页：** <https://rustgis.app/>
- **入库日期：** 2026-10-21
- **一句话说明：** 以 Rust 编写的跨平台桌面与浏览器 GIS，用于地理空间数据制图、编辑与分析；支持 CLI、Python 与可扩展插件。
- **沉淀到 wiki：** 是 → [`wiki/entities/rustgis.md`](../../wiki/entities/rustgis.md)

## 开源状态核查（2026-10-21）

| 项 | 值 |
|----|-----|
| **开放程度** | **已开源** — 官方 GitHub 仓库公开，包含桌面端、WebAssembly 浏览器端及 CLI 等实现 |
| **主要语言** | Rust（Cargo workspace；含 `crates/` 与 `apps/`） |
| **许可** | Apache-2.0（仓库元数据与 `Cargo.toml`） |
| **平台** | Windows、macOS、Linux 与浏览器 |
| **官方项目页** | <https://rustgis.app/> |

## README 摘要

RustGIS 将 GIS 工作台、分析工具和地图发布能力整合在一款以 Rust 实现的应用中。官方 README 列出的数据入口包括 GeoJSON、CSV/TSV、GeoPackage、GeoParquet、GeoTIFF/COG、STAC 与公开数据门户，也支持 PostGIS、MySQL 等在线数据源。地图侧包含投影、三维地球、地形和要素拉伸；分析侧集成 GeoLibre 与 Whitebox 工具。

交互操作可通过 `rustgis-cli` 和 Python 包脚本化；项目还提供 AI Assistant、MCP Server 及 WebAssembly 插件机制。AI 助手使用用户选择的模型执行 RustGIS 命令；项目页说明数据默认留在本机，联网数据源、插件和模型调用按用户操作发生。

## 与机器人研究/工程的关联点

- 可作为地理空间数据的桌面检查与分析工具，用于查看带坐标的环境、地形或地图数据，并导出制图成果。
- 对机器人导航而言，它位于**数据准备/可视化层**；仓库资料没有把它描述为 SLAM、实时定位、ROS 2 costmap 或机器人导航运行时。
- 当输入来自机器人系统时，应另行确认坐标参考系、时间戳、地图栅格/点云转换与在线数据链路是否满足目标流程，不能把通用 GIS 格式支持等同于机器人消息接口支持。

## 对 wiki 的映射

- 实体页：[wiki/entities/rustgis.md](../../wiki/entities/rustgis.md)
- 导航与 SLAM 栈总览：[wiki/overview/navigation-slam-autonomy-stack.md](../../wiki/overview/navigation-slam-autonomy-stack.md)
- 项目页归档：[sources/sites/rustgis-app.md](../sites/rustgis-app.md)

## 参考链接

- GitHub 仓库：<https://github.com/opengeos/RustGIS>
- 官网与文档：<https://rustgis.app/>
- 浏览器版：<https://web.rustgis.app>
