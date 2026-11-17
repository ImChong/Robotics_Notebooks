# Tidy Design（tidydesigns/tidy）

> 官方仓库来源归档

- **类型：** repo
- **来源：** Tidy Designs（GitHub）
- **链接：** https://github.com/tidydesigns/tidy
- **官网：** https://tidydesign.co
- **许可：** Apache-2.0（GitHub 仓库元数据）
- **入库日期：** 2026-10-19
- **复核日期：** 2026-11-17
- **原文版本：** `tidydesigns/tidy` 官方 `main`，2026-11-17 访问版本；实现与发行渠道会继续变化
- **核对范围：** 已通读官方 `README.md` 全文（定位、Agentic development/design、桌面启动、本地 Web 开发入口、贡献、安全与许可）、`docs/development.md` 全文（Web 本地开发、扩展、MCP、Cloudflare 自托管、安全边界）及 `apps/gpt-plugin/README.md` 全文（插件 UI、MCP 关联、预览与部署）；重点阅读 `apps/desktop/README.md` 的项目线程、Harness 权限和确认、设计文件/线程存储与桌面启动章节（其余设备/云同步等章节未逐行核读）；并对照根 `package.json`、`CONTRIBUTING.md` 的脚本与依赖。以上核对为文档阅读，未声称运行或全面审计代码。
- **通读：** 已完成
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
- **本地桌面：** 当前 README 指定 Bun 1.3.14、Node.js 22.12+，执行 `bun install --frozen-lockfile && bun dev:desktop`；本地项目与编码线程**不需要** Web app、数据库或云账号。
- **Web 自托管：** `docs/development.md` 使用 Docker Compose/PostgreSQL、`apps/web/.env.local` 和 `bun dev`；数据库测试另需 PostgreSQL 17 命令行工具。
- **自托管：** Web 版本可部署于 Cloudflare Workers，配合 PostgreSQL/Hyperdrive、R2 和 Durable Objects；必须独立配置密钥、账户、授权与数据库权限，桌面本地运行不要求这些基础设施。
- **CLI 与权限：** 桌面线程通过本机已安装并认证的 Codex / Claude Code 等 CLI；Codex 使用 App Server（stdio），Claude Code 使用双向流式 JSON；需要用户审批时按照原生权限模型处理，不能默认全权限。
- **设计文件：** 桌面项目将可编辑设计与对话线程关联，Agent 与画布共享设计文件；本地项目的 `.tidy/designs/<id>.json` 可版本化，私有线程/会话资料受到不同存储与授权约束。

仓库 README 未声称提供机器人专用画布、URDF/MJCF 编辑或仿真集成；不应从「面向工程师」推断为现成功能。

## 对 wiki 的映射

- 主实体：[Tidy Design](../../wiki/entities/tidy-design.md)
- MCP 概念：[Model Context Protocol](../../wiki/concepts/model-context-protocol.md)
- Coding agent 工程背景：[Agentic Coding 时代的软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)
- 官网归档：[Tidy Design 官网](../sites/tidy-design.md)

## 原文依据与适用边界

- [README：产品、桌面功能与安装](https://github.com/tidydesigns/tidy/blob/main/README.md)
- [Desktop guide：本地线程、代理协议与权限、设计文件](https://github.com/tidydesigns/tidy/blob/main/apps/desktop/README.md)
- [Web self-hosting：数据库、MCP 与部署](https://github.com/tidydesigns/tidy/blob/main/docs/development.md)
- [GPT plugin：MCP Apps 预览与安装](https://github.com/tidydesigns/tidy/blob/main/apps/gpt-plugin/README.md)
- [Root package scripts](https://github.com/tidydesigns/tidy/blob/main/package.json)
- **开源范围：** 官方仓库 Apache-2.0，附带第三方资产各自许可；云端协作与个人的本地 coding threads 具有不同权限、存储与账号前提，不能当作一套无差别部署方案。
- **复现状态：** 未启动 Electron/Web、未进行数据库或模型联调；功能事实来自上述官方原文和公开代码目录，不能当作端到端验证结果。

## 参考链接

- 官方仓库：<https://github.com/tidydesigns/tidy>
- README：<https://github.com/tidydesigns/tidy/blob/main/README.md>
- 官网：<https://tidydesign.co>
- 应用：<https://app.tidydesign.co>
