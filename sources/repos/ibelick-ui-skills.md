# UI Skills（ibelick/ui-skills）

> 来源归档

- **标题：** UI Skills — Skills for Design Engineers
- **类型：** repo
- **作者 / 维护者：** ibelick（仓库账号）
- **链接：** https://github.com/ibelick/ui-skills
- **官方站：** https://www.ui-skills.com/
- **入库日期：** 2026-10-07
- **协议：** MIT（仓库 package metadata）
- **核查版本：** package.json 为 0.2.4（核查日期：2026-10-07）
- **一句话说明：** 面向设计工程的策展型 Agent Skills 目录，提供网站、CLI、技能内容 URL 和 MCP 工具，供编码代理发现与读取 UI 技能。
- **为什么值得保留：** 将 UI / 前端设计指导作为可路由的 Markdown skills 提供，目录记录发布者、上游仓库、内容地址和主题；便于与通用发现工具、单项前端技能及全生命周期工程技能库比较。
- **沉淀到 wiki：** 是 → [UI Skills（ibelick/ui-skills）](../../wiki/entities/ibelick-ui-skills.md)

## 仓库要点（归纳）

- **目录数据：** `src/data/registry.ts` 定义技能项字段：slug、pathSlug、发布者 / 仓库、rawUrl、githubUrl、name、description、topics。
- **主题：** 目录覆盖可访问性、动效、系统、视觉、交互、性能、前端、框架和工具等主题。
- **CLI：** `bin/ui-skills.ts` 提供 `start`、`categories`、`list [--category <topic>]`、`get <slug>`。CLI 读取 `/skills/registry.json`，并从站点内容路径读取技能正文。
- **MCP：** `src/lib/mcp-server.ts` 提供 `list_skills` 与 `get_skill`；过滤字段包括 slug、pathSlug、名称和描述。
- **统一内容路径：** `src/lib/agent-skills-discovery.ts` 说明 CLI、MCP 与 skills 目录使用兼容的 pathSlug / 内容 URL。服务端可优先读取部分同仓技能；其它发布者内容按 registry 的远程地址加载。
- **服务和测试：** 项目由 Astro / TypeScript 实现；package scripts 包含 typecheck、registry skill 检查、测试、构建与 smoke 检查。仓库将这些实现检查串为 `npm run check`。
- **目录维护：** 项目附有面向技能发布者的提交与校验约定；被收录条目的归属和许可证应按各上游仓库分别核对。

## 上游入口

- 官方站：<https://www.ui-skills.com/>
- 仓库 README：<https://github.com/ibelick/ui-skills#readme>
- CLI：<https://github.com/ibelick/ui-skills/blob/main/bin/ui-skills.ts>
- registry：<https://github.com/ibelick/ui-skills/blob/main/src/data/registry.ts>
- MCP 服务：<https://github.com/ibelick/ui-skills/blob/main/src/lib/mcp-server.ts>
- 项目页核查： [UI Skills 官方站归档](../sites/ui-skills.md)
