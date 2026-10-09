## [2026-10-09] ingest | REA 软件逆向调查工具 — 归档官方源码、项目页并提炼 MCP/CLI 工程边界

- 意图：将 morluto/rea 收录为单一工具实体，解释它如何把本机软件分析工具接入编码代理。
- 开源结论：GitHub 主仓库为 MIT 开源；深度原生分析依赖 Hopper、Ghidra 或 IDA 等外部 provider，REA 不等于反编译引擎。
- 关键说明：区分静态分析和运行时采集，并记录本机分析、Agent/model provider 数据边界及目标授权责任。
