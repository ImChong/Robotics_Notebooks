# Part-DB

> 来源归档

- **标题：** Part-DB — Electronic component inventory management
- **类型：** repo
- **代码：** <https://github.com/Part-DB/Part-DB-server>（AGPL-3.0-or-later）
- **文档：** <https://docs.part-db.de/>
- **入库日期：** 2026-10-27
- **一句话说明：** 自托管电子元件库存系统，支持分类、参数、库存数量和存放位置管理。
- **开源状态：** 官方服务端源码公开，采用 GNU AGPL v3 或更高版本。
- **沉淀到 wiki：** [Part-DB](../../wiki/entities/part-db.md)

## 项目要点

- 服务部署在 Web 服务器上，用户通过浏览器访问；每个元件可关联分类、封装、制造商、多个存放位置、价格及数据手册。
- 支持参数搜索、条码标签、用户权限、库存变更记录和数据导入导出。
- 支持 KiCad BOM 导入及 KiCad 数据源集成；支持 SQLite、MySQL 和 PostgreSQL。
- 官方提供 Docker 镜像和安装文档；仓库中的服务端是旧版 Part-DB 的重写版本。

## 官方入口

- [GitHub 仓库及 README](https://github.com/Part-DB/Part-DB-server)
- [安装文档](https://docs.part-db.de/installation/)
- [KiCad 集成文档](https://docs.part-db.de/usage/eda_integration.html)
