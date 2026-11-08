# Pascal Editor 官方站 / 开发者文档

- **类型：** site / 官方产品文档
- **项目主页：** https://editor.pascal.app
- **文档：** https://editor.pascal.app/docs/developers/local-editor
- **MCP 文档：** https://editor.pascal.app/docs/developers/mcp
- **代码：** https://github.com/pascalorg/editor
- **核对日期：** 2026-11-08
- **核对范围：** 核对本地 CLI 安装章节、MCP 连接方式和托管/本地的存储与鉴权边界；网页剩余全部页面未逐篇阅读。
- **通读：** 未完成
- **源码开放结论：** 根 README 指向开源仓库及 MIT 许可；本地 CLI+MCP 可以不通过 Pascal 云端账号；托管服务需其账号授权，商业服务条款另行适用。
- **映射：** [仓库归档](../repos/pascalorg-editor.md) → [项目实体](../../wiki/entities/pascal-editor.md)

## 核心核查

官方文档明确 `npx @pascal-app/cli editor` 与 `pascal mcp connect` 属于本地安装/连接流程；`https://editor.pascal.app/api/mcp` 是另一个托管入口。公开文档的 Node 最小版本可能随 CLI 发布版本改变，复现时以 npm 包 engines 与目标版本为准。
