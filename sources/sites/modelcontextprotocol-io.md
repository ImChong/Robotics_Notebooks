# Model Context Protocol 官方文档（modelcontextprotocol.io）

> 来源归档

- **标题：** Model Context Protocol — Official Documentation
- **类型：** site（官方规范 / 开发者文档）
- **主体：** Model Context Protocol 项目（最初由 Anthropic 开源；规范与文档托管于本站与 GitHub org）
- **入口：** <https://modelcontextprotocol.io>
- **文档索引（LLM）：** <https://modelcontextprotocol.io/llms.txt>
- **入库日期：** 2026-08-03
- **一句话说明：** MCP 的 **权威工程入口**：Getting Started、Architecture、Server/Client Concepts、多语言 SDK、Inspector、Registry、Extensions，以及按日期版本化的 **Specification**（TypeScript schema 为真源）。
- **开源状态：** **已开源** — 规范 / schema / 文档见 [modelcontextprotocol/modelcontextprotocol](../repos/modelcontextprotocol.md)；参考 servers、SDKs、Inspector 等见同组织公开仓。
- **沉淀到 wiki：** [Model Context Protocol](../../wiki/concepts/model-context-protocol.md)

---

## 抓取说明

- 以 **2026-08-03** 对下列一手页面的 Markdown 镜像抓取为准（Mintlify `.md` 路径）：
  - [What is MCP?](https://modelcontextprotocol.io/docs/2026-07-28/getting-started/intro.md)
  - [Architecture overview](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture.md)
  - [Understanding MCP servers](https://modelcontextprotocol.io/docs/2026-07-28/learn/server-concepts.md)
  - [Understanding MCP clients](https://modelcontextprotocol.io/docs/2026-07-28/learn/client-concepts.md)
  - [Specification 2026-07-28 index](https://modelcontextprotocol.io/specification/2026-07-28/index.md)
  - [Basic / Transports](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports.md)
- **当前文档树版本前缀：** `docs/2026-07-28/` 与 `specification/2026-07-28/`（schema 目录另有历史版本 `2024-11-05` … `2025-11-25`）。
- 协议演进快；引用 wire 行为时务必标注 **protocolVersion**，勿混用 `initialize` 时代与 `2026-07-28` 无状态核心。

---

## 一句话（官方 Intro）

MCP 是连接 **AI 应用** 与 **外部系统** 的开源标准——像「AI 应用的 USB-C」：统一对接数据源、工具与工作流（专用 prompts）。

---

## 为什么值得保留

- 本库大量实体（FreeCAD MCP、Draw.io MCP、DimOS MCP、UE/Unity MCP 方向）都 **依赖本协议语义**；此前只有应用桥归档，缺少协议层一手源。
- Intro + Architecture 给出 Host / Client / Server、Tools / Resources / Prompts、Stdio vs Streamable HTTP 的 **规范用语**，可校正 wiki 中口语化混用。
- `2026-07-28` 规范对 **无状态核心、`server/discover`、MRTR、Extensions** 有重大修订——选型与兼容性必须以本站为准。

---

## 架构摘要（Architecture，2026-07-28 文档）

### 参与者

| 角色 | 定义 |
|------|------|
| **MCP Host** | AI 应用（如 Claude Code / Claude Desktop），协调一个或多个 MCP Client |
| **MCP Client** | Host 内组件；与 **一个** MCP Server 维持专用连接 |
| **MCP Server** | 向 Client 提供上下文 / 能力的程序 |

- **Stdio** 本地 server：通常一对一服务单个 Client。
- **Streamable HTTP** 远程 server：通常服务多个 Client。

### 两层

| 层 | 内容 |
|----|------|
| **Data layer** | JSON-RPC 2.0 消息、能力/版本发现、Tools / Resources / Prompts、通知等 |
| **Transport layer** | 连接建立、消息成帧、认证；与数据层解耦 |

### 标准传输（仅两种官方）

| 传输 | 机制 |
|------|------|
| **stdio** | 本机进程 stdin/stdout；换行分隔 JSON-RPC；无网络开销 |
| **Streamable HTTP** | HTTP POST（Client→Server）+ 可选 SSE；支持 bearer / API key / 自定义头；推荐 OAuth 取 token |

自定义传输允许，但 **必须** 保留 JSON-RPC 消息格式、消息模式与 per-request metadata；官方本周期 **不再新增** 其他标准传输（见 Roadmap）。

### Server Primitives

| Primitive | 谁控制 | 协议操作（概念） |
|-----------|--------|------------------|
| **Tools** | Model 决定何时调用 | `tools/list`、`tools/call` |
| **Resources** | Application 拉取只读上下文 | URI 寻址的数据 |
| **Prompts** | User 选用模板 | 可复用 prompt 模板 |

### `2026-07-28` 数据层要点（相对旧版）

- **无状态：** 每个请求在 `_meta` 携带 `io.modelcontextprotocol/protocolVersion` 与相关 capabilities；server 不依赖会话状态推断。
- **发现：** 强制能力面通过 **`server/discover`**（可缓存）；旧版 `initialize` / `initialized` 握手进入兼容路径。
- **Sampling** 等能力在该版本文档中标记为 **deprecated**（见规范 Deprecated 节）。
- **Client features：** 如 Elicitation（`elicitation/create`）等，让 server 可向用户要确认/补充输入。

### 项目范围（官方明确）

包含：Specification、SDKs、Inspector 等开发工具、Reference Servers。  
**不包含：** 规定 AI 应用如何使用 LLM 或如何管理已提供的上下文。

---

## 生态入口（同站）

| 入口 | URL |
|------|-----|
| Spec latest / 版本化 | <https://modelcontextprotocol.io/specification/latest> |
| SDKs | <https://modelcontextprotocol.io/docs/2026-07-28/sdk> |
| Inspector | <https://modelcontextprotocol.io/docs/2026-07-28/tools/inspector> |
| Registry | <https://modelcontextprotocol.io/registry/about> |
| Extensions（Apps / Tasks / Auth…） | <https://modelcontextprotocol.io/extensions/overview> |
| Roadmap | <https://modelcontextprotocol.io/development/roadmap.md> |

---

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| MCP 协议概念 | `wiki/concepts/model-context-protocol.md` |
| RPC 下层对照 | `wiki/concepts/remote-procedure-call.md` |
| 应用桥样本 | `wiki/entities/freecad-mcp.md` 等 |

## 参考链接

- <https://modelcontextprotocol.io>
- <https://modelcontextprotocol.io/llms.txt>
- <https://modelcontextprotocol.io/docs/2026-07-28/getting-started/intro>
- <https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture>
- <https://modelcontextprotocol.io/specification/2026-07-28>
- Anthropic 发布公告归档：[anthropic-model-context-protocol.md](./anthropic-model-context-protocol.md)
