# Introducing the Model Context Protocol（Anthropic 公告）

> 来源归档

- **标题：** Introducing the Model Context Protocol
- **类型：** site / blog（官方发布公告）
- **来源：** Anthropic（人类智能）
- **链接：** https://www.anthropic.com/news/model-context-protocol
- **发布日期：** 2024-11-25
- **入库日期：** 2026-08-03
- **一句话说明：** Anthropic 宣布 **开源 Model Context Protocol（MCP）**——用统一开放标准连接 AI 助手与内容库、业务工具与开发环境，替代「每个数据源一套自定义集成」的碎片化模式。
- **开源状态：** **已开源** — 公告同时推出规范与 SDK、Claude Desktop 本地 MCP server 支持，以及预构建 MCP servers 开源仓；协议后续演进见 [modelcontextprotocol.io](./modelcontextprotocol-io.md) 与 [spec 仓库](../repos/modelcontextprotocol.md)。
- **沉淀到 wiki：** [Model Context Protocol](../../wiki/concepts/model-context-protocol.md)

---

## 抓取说明

- 以 **2026-08-03** 对 Anthropic News 页公开 HTML 正文抽取为准。
- 本页是 **协议诞生的一手公告**；工程细节（传输、Primitives、版本）以官方文档与规范为准，勿仅凭本公告推断当前 wire 行为。

---

## 公告要点（2024-11-25）

| 主题 | 摘要 |
|------|------|
| **问题** | 模型被信息孤岛与遗留系统隔离；每个新数据源都要单独定制集成，难以规模化 |
| **方案** | MCP：通用开放标准，连接 AI 系统与数据源；开发者可写 **MCP server** 暴露数据，或写 **MCP client**（AI 应用）去连接 |
| **当日三件套** | ① 规范与 SDK；② Claude Desktop 本地 MCP server 支持；③ 开源 MCP servers 仓库 |
| **预构建示例** | Google Drive、Slack、GitHub、Git、Postgres、Puppeteer 等 |
| **早期采用** | Block、Apollo；开发工具侧 Zed、Replit、Codeium、Sourcegraph 等 |
| **作者** | David Soria Parra、Justin Spahr-Summers（于 Anthropic 创建） |
| **定位** | 协作开源项目与生态，而非闭源厂商私有插件协议 |

### 架构一句话（公告原文语义）

开发者既可以 **通过 MCP servers 暴露数据**，也可以 **构建连接这些 servers 的 AI 应用（MCP clients）**——双向、标准化。

---

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| MCP 协议概念与生态边界 | `wiki/concepts/model-context-protocol.md` |
| 与 RPC 概念对照 | `wiki/concepts/remote-procedure-call.md` |
| 机器人栈中的 MCP 应用样本 | `wiki/entities/freecad-mcp.md`、`wiki/entities/drawio-scientific-illustrator.md`、`wiki/entities/dimensionalos-dimos.md` 等 |

## 参考链接

- <https://www.anthropic.com/news/model-context-protocol>
- <https://modelcontextprotocol.io>
- <https://github.com/modelcontextprotocol>
