# Octop

- **类型：** repo
- **链接：** <https://github.com/TencentCloud/Octop>
- **许可：** MIT
- **核查版本：** `4811eb45996169e86006db1b6451264ff4758223`，2026-11-05。
- **开源状态：** 已开放服务端、控制台源码、安装脚本和测试；模型服务需另行配置。
- **主要资料：** [README](https://github.com/TencentCloud/Octop/blob/4811eb45996169e86006db1b6451264ff4758223/README.md)、[架构文档](https://github.com/TencentCloud/Octop/blob/4811eb45996169e86006db1b6451264ff4758223/docs/architecture.md)。
- **入口：** `octop run` → `src/octop/cli/commands/run.py` → `src/octop/launch.py` → `OctopServer.start()`。
- **知识页：** [Octop](../../wiki/entities/tencent-octop.md)。

Octop 将控制台、即时通讯通道、定时任务和多个专家 Agent 组合为自托管服务；默认使用 SQLite，也支持 PostgreSQL。已通读上述 README 与架构文档，并核查启动入口及网关组装源码；尚未独立部署或验证其全部组件和安全声明。
