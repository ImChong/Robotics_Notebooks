# RustGIS 官方项目页（rustgis.app）

- **类型：** 网站 / GIS 产品文档
- **入口：** <https://rustgis.app/>
- **主体：** opengeos；项目名 RustGIS
- **代码：** <https://github.com/opengeos/RustGIS>（已开源，见[仓库归档](../repos/rustgis.md)）
- **浏览器版：** <https://web.rustgis.app>
- **入库日期：** 2026-10-21
- **核查依据：** 官方首页、导航文档与仓库 README；核查日期 2026-10-21

## 一句话

**RustGIS** 是以 Rust 编写的跨平台 GIS 应用，用于地图制图、地理数据编辑与分析，提供桌面版和 WebAssembly 浏览器版；交互命令可由 CLI、Python 包或 AI Assistant 调用。

## 公开功能要点

| 模块 | 官方资料描述 |
|------|--------------|
| 数据 | GeoJSON、Shapefile、GeoPackage、GeoParquet、GeoTIFF 等；连接 PostGIS、MySQL；浏览 STAC、Overture Maps 和开放数据门户 |
| 制图与地图 | 符号化、标注、投影、三维地球、地形、页面布局和地图系列 |
| 分析 | SQL 选择、叠加/邻近分析、ModelBuilder，以及 GeoLibre、Whitebox 地形、水文和 LiDAR 工具 |
| 编辑 | 要素和顶点编辑、吸附、属性规则、拓扑检查与撤销 |
| 自动化 | `rustgis-cli`、Python 包；命令也可由 AI Assistant 调用 |
| 扩展 | WebAssembly 插件；官方文档提供 MCP Server 入口 |
| 平台 | Windows、macOS、Linux 原生应用与浏览器应用 |

## 开源与边界（2026-10-21）

- **代码：** 已开源；项目页直接链接至公开 GitHub 仓库。
- **许可：** 仓库元数据与 Cargo workspace 声明 Apache-2.0。
- **数据本地性：** 官方首页称数据默认保留在本机；联网地图服务、插件和用户选定的 AI 模型会按需联网。
- **机器人适用边界：** 可用于地理空间数据浏览、分析和制图；项目文档没有声称提供机器人实时定位、SLAM 或 ROS 2 消息集成。

## 对 wiki 的映射

- [RustGIS 实体页](../../wiki/entities/rustgis.md)
- [导航·SLAM·自动驾驶开源栈总览](../../wiki/overview/navigation-slam-autonomy-stack.md)
- [仓库归档](../repos/rustgis.md)

## 参考链接

- 官方首页与导航：<https://rustgis.app/>
- 官方代码：<https://github.com/opengeos/RustGIS>
- 桌面端与 Web 端入口：<https://web.rustgis.app>
