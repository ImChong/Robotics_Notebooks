# Tidy Design（tidydesigns/tidy）

> 官方仓库来源归档

- **类型：** repo
- **来源：** Tidy Designs（GitHub）
- **链接：** https://github.com/tidydesigns/tidy
- **官网：** https://tidydesign.co
- **许可：** Apache-2.0（GitHub 仓库元数据）
- **入库日期：** 2026-10-19
- **一句话说明：** 开源协作式产品设计画布，通过 MCP 让 agent 读取、创建和编辑可继续修改的界面设计。
- **为什么值得保留：** 为 coding agent 提供可编辑的设计工作区，展示自然语言改动、画布审阅、继续迭代的协作模式。机器人团队可探索将其用于界面原型，但官方仓库未展示经验证的机器人专用流程。
- **沉淀到 wiki：** 是 → [wiki/entities/tidy-design.md](../../wiki/entities/tidy-design.md)

## 仓库概况（官方 README 与 GitHub 仓库元数据，2026-10-19）

- 默认分支：main
- 主要语言：TypeScript
- 许可证：Apache-2.0
- 仓库介绍：面向设计师、工程师和 agent 的开源产品画布

## README 摘要

- **可编辑画布：** 人在画布上创建界面，图层保持可编辑。
- **Agent 接入：** 通过 MCP 读取、创建和编辑设计；仓库提供 agent plugin 配置指南及 native runner 指南。
- **协作与采集：** 支持同一工作区协作，并提供浏览器扩展把网页带入 Tidy。
- **本地开发：** README 指定 Bun 1.3.14、Git 和 Docker Compose；数据库测试另需 PostgreSQL 17 命令行工具。
- **自托管：** 提供 Cloudflare Workers、PostgreSQL 与 R2 等部署说明，部署者需自行配置凭证、数据库与存储绑定。

仓库 README 未声称提供机器人专用画布、URDF/MJCF 编辑或仿真集成；不应从「面向工程师」推断为现成功能。

## 对 wiki 的映射

- 主实体：[Tidy Design](../../wiki/entities/tidy-design.md)
- MCP 概念：[Model Context Protocol](../../wiki/concepts/model-context-protocol.md)
- Coding agent 工程背景：[Agentic Coding 时代的软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)
- 官网归档：[Tidy Design 官网](../sites/tidy-design.md)

## 参考链接

- 官方仓库：<https://github.com/tidydesigns/tidy>
- README：<https://github.com/tidydesigns/tidy/blob/main/README.md>
- 官网：<https://tidydesign.co>
- 应用：<https://app.tidydesign.co>
