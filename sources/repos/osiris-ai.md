# OSIRIS — Open Source Intelligence & Reconnaissance Integrated System

> 来源归档

- **标题：** OSIRIS
- **类型：** repo
- **维护者：** simplifaisoul
- **代码：** https://github.com/simplifaisoul/osiris
- **演示：** https://osirisai.live/
- **文档：** https://www.osirisai.live/docs
- **许可：** MIT（以仓库 LICENSE 为准）
- **入库日期：** 2026-10-04
- **一句话说明：** 将航班、卫星、公开摄像头、地震、火点、天气、冲突与网络威胁等多源信息叠加到 WebGL 全球地图的 OSINT 仪表盘。
- **项目页归档：** [OSIRIS 官网与文档](../sites/osirisai-live.md)
- **沉淀到 wiki：** [OSIRIS 全球 OSINT 情报地图](../../wiki/entities/osiris-global-osint.md)

## 开源状态

- **已开源：** 官方 GitHub 仓库公开，仓库标注 MIT 许可，包含应用源码、Docker 部署相关文件及环境变量示例。
- **技术栈：** README 标注 Next.js App Router、TypeScript、MapLibre GL JS/WebGL；细节应以当前主分支代码为准。
- **边界：** 网站是多种公开数据源的聚合前端；数据质量、可用性、时间戳与授权由对应上游提供方决定。README 将海事数据列作静态港口/航道情报，而 API 文档列有 vessel positions 接口，因此不把全部海事目标描述成已核实的实时 AIS 跟踪。

## 对 wiki 的映射

- [OSIRIS 全球 OSINT 情报地图](../../wiki/entities/osiris-global-osint.md) — 系统结构、数据层、地图仪表盘设计经验与局限。
- [官网、演示与 API 文档](../sites/osirisai-live.md) — 项目网页、数据与隐私说明的核验入口。
