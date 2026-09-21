# next-ai-draw-io（DayuanJiang/next-ai-draw-io）

> 来源归档

- **标题：** Next AI Draw.io — AI-Powered Diagram Creation Tool
- **类型：** repo（Next.js Web 应用 + MCP Server npm 包 + 桌面 Release）
- **作者：** DayuanJiang
- **链接：** https://github.com/DayuanJiang/next-ai-draw-io
- **项目页：** https://next-ai-drawio.jiang.jp/
- **克隆：** `https://github.com/DayuanJiang/next-ai-draw-io.git`
- **许可：** Apache-2.0
- **入库日期：** 2026-09-21
- **一句话说明：** Next.js 聊天界面 + **react-drawio** 画布，经 **Vercel AI SDK** 多 provider 流式生成/修改 **draw.io XML**；另发 **`@next-ai-drawio/mcp-server`**，嵌入 HTTP 服务在浏览器实时预览，供 Cursor / Claude Desktop / VS Code 等 MCP 客户端建图。
- **开源状态：** **已开源** — Apache-2.0；Web 源码、MCP 包、Docker/桌面 Release 均可获取。项目页 [`sources/sites/next-ai-drawio-jiang-jp.md`](../sites/next-ai-drawio-jiang-jp.md) 链回 GitHub。
- **沉淀到 wiki：** 是 → [`wiki/entities/next-ai-draw-io.md`](../../wiki/entities/next-ai-draw-io.md)

## 仓库概况（2026-09-21 GitHub API / README）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`DayuanJiang/next-ai-draw-io`） |
| 默认分支 | `main` |
| 主要语言 | TypeScript |
| Stars / Forks | ~36.0k / ~3.8k |
| 描述 | Next.js web app integrating AI with draw.io diagrams via natural language |
| Topics | `ai`, `diagrams`, `productivity` |
| 前端栈 | Next.js 16.x、React 19.x |
| MCP 包 | `@next-ai-drawio/mcp-server`（`packages/mcp-server/`） |
| 本地 dev 端口 | `6002`（与 MCP 嵌入 HTTP 默认同端口） |

## 为何值得保留

- **高星 AI 制图闭环样本：** 自然语言 → LLM 生成 **draw.io XML** → 浏览器画布实时渲染 → 版本历史 / 导出 `.drawio`/PNG/SVG；与本站 wiki/roadmap 的 Mermaid 静态图互补，适合 **云架构、流程图、论文机制图** 的快速迭代。
- **MCP 一等公民：** 与 [Draw.io Scientific Illustrator](../../wiki/entities/drawio-scientific-illustrator.md)（Codex 桌面 live MCP）形成对照——本仓 **Web 嵌入 + npm MCP**，Cursor/Claude Code 一行 `npx @next-ai-drawio/mcp-server@latest` 即可在浏览器预览。
- **多 provider 与自托管：** AWS Bedrock（默认）、OpenAI、Anthropic、Google、Azure、Ollama、DeepSeek、OpenRouter 等；`AI_MODELS_CONFIG` / Admin Panel（`/admin`）支持服务端多模型与配额；Docker / Vercel / EdgeOne / Cloudflare 文档齐全。
- **与 Archify / Diagram Design 分工：** [Archify](../../wiki/entities/archify.md) 产出 **校验 JSON→HTML**；[Diagram Design](../../wiki/entities/diagram-design.md) 产出 **editorial HTML/SVG**；本仓产出 **原生 `.drawio` 生态** 的可编辑 XML。

## README / MCP 要点（归纳）

### Web 应用

- **交互：** 聊天 refine 图；上传图片/PDF/文本复刻；云架构模板；动画连接器；diagram history 回滚。
- **技术：** Next.js 路由 + **Vercel AI SDK**（`ai` + `@ai-sdk/*`）流式响应 + **react-drawio** 操纵 XML。
- **模型建议：** 需长格式、严格格式约束（draw.io XML）；推荐 Claude Sonnet 4.5、GPT-5.1、Gemini 3 Pro、DeepSeek V3.2/R1；Claude 系列对 AWS/Azure/GCP 云图标训练较好。
- **安装：** `git clone` → `npm install` → `cp env.example .env.local` → `npm run dev` → `http://localhost:6002`。
- **桌面：** GitHub Releases（Win/macOS/Linux）。

### MCP Server（`@next-ai-drawio/mcp-server`）

- **配置示例：** `"command": "npx", "args": ["@next-ai-drawio/mcp-server@latest"]`；Claude Code：`claude mcp add drawio -- npx @next-ai-drawio/mcp-server@latest`。
- **架构：** MCP Client（stdio）↔ 嵌入 HTTP Server（默认 `:6002`）↔ 浏览器 draw.io embed（默认 `embed.diagrams.net`，可 `DRAWIO_BASE_URL` 自建）。
- **工具：** `start_session`、`create_new_diagram`（XML）、`load_diagram`、`edit_diagram`（按 cell id 增删改）、`get_diagram`、`export_diagram`、多 page 管理（`list_pages` / `add_page` / `rename_page` / `delete_page`）。
- **私有部署：** 可指向自托管 draw.io Docker（`jgraph/drawio`）。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 实体页（主） | [`wiki/entities/next-ai-draw-io.md`](../../wiki/entities/next-ai-draw-io.md) |
| 项目页归档 | [`sources/sites/next-ai-drawio-jiang-jp.md`](../sites/next-ai-drawio-jiang-jp.md) |
| 桌面 live MCP 对照 | [`wiki/entities/drawio-scientific-illustrator.md`](../../wiki/entities/drawio-scientific-illustrator.md) |
| 校验 HTML 系统图对照 | [`wiki/entities/archify.md`](../../wiki/entities/archify.md) |
| editorial 图 Skill 对照 | [`wiki/entities/diagram-design.md`](../../wiki/entities/diagram-design.md) |
| MCP 概念 | [`wiki/concepts/model-context-protocol.md`](../../wiki/concepts/model-context-protocol.md) |

## 参考链接

- 源码仓库：<https://github.com/DayuanJiang/next-ai-draw-io>
- 在线演示：<https://next-ai-drawio.jiang.jp/>
- MCP README：`packages/mcp-server/README.md`
- Provider 配置：`docs/en/ai-providers.md`
- Admin Panel：`docs/en/admin-panel.md`
