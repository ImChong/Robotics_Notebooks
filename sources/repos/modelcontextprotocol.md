# Model Context Protocol（规范与文档仓库）

- **类型：** repo（规范 + schema + 官方文档源）
- **标题：** modelcontextprotocol/modelcontextprotocol
- **组织：** [github.com/modelcontextprotocol](https://github.com/modelcontextprotocol)
- **链接：** https://github.com/modelcontextprotocol/modelcontextprotocol
- **主页：** https://modelcontextprotocol.io
- **入库日期：** 2026-08-03
- **一句话说明：** MCP 的 **规范与文档真源仓**：TypeScript schema（`schema/<YYYY-MM-DD>/schema.ts`）为一手协议定义，并生成 JSON Schema；Mintlify 文档构建后发布到 modelcontextprotocol.io。
- **开源状态：** **已开源** — 仓库公开；LICENSE 注明项目正从 **MIT → Apache-2.0** 过渡（新贡献 Apache-2.0；文档贡献多为 CC-BY-4.0；未同意再授权的历史 MIT 贡献仍按 MIT）。
- **关联站点：** [modelcontextprotocol-io.md](../sites/modelcontextprotocol-io.md)、[Anthropic 公告](../sites/anthropic-model-context-protocol.md)
- **沉淀到 wiki：** [Model Context Protocol](../../wiki/concepts/model-context-protocol.md)

---

## 仓库元数据（2026-08-03）

| 字段 | 值 |
|------|-----|
| 描述 | Specification and documentation for the Model Context Protocol |
| Stars（约） | 8.8k |
| 默认分支 | `main` |
| 作者（README） | David Soria Parra（@dsp）、Justin Spahr-Summers（@jspahrsummers） |

## Schema 版本目录（`schema/`，API 列举）

| 目录 | 备注 |
|------|------|
| `2024-11-05` | 早期版本（贴近 Anthropic 首发窗口） |
| `2025-03-26` | 历史修订 |
| `2025-06-18` | 历史修订 |
| `2025-11-25` | 常见「稳定」引用点之一 |
| `2026-07-28` | 文档树当前主推版本（无状态核心等） |
| `draft` | 草稿 |

真源文件形态：`schema/<version>/schema.ts`（及生成的 `schema.json`）。

## 同组织关键公开仓（一手生态，2026-08-03 API）

| 仓库 | Stars（约） | 角色 |
|------|------------|------|
| [servers](https://github.com/modelcontextprotocol/servers) | ~89k | 参考 / 示例 MCP servers |
| [python-sdk](https://github.com/modelcontextprotocol/python-sdk) | ~24k | 官方 Python SDK |
| [typescript-sdk](https://github.com/modelcontextprotocol/typescript-sdk) | ~13k | 官方 TypeScript SDK |
| [inspector](https://github.com/modelcontextprotocol/inspector) | ~11k | 官方调试客户端（Web / CLI / TUI） |
| [registry](https://github.com/modelcontextprotocol/registry) | ~7k | 社区驱动 MCP server 注册服务 |
| go / csharp / rust / java / … SDKs | — | 官方多语言 SDK |

> 本归档 **不** 把每个 SDK 拆成独立 sources 页；需要时从本表深挖。机器人栈应用桥（FreeCAD / DimOS / Draw.io 等）另有各自 `sources/repos/`。

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| MCP 协议与版本 | `wiki/concepts/model-context-protocol.md` |
| Inspector 调试实践 | 链到概念页「工程实践」；官方文档 [Inspector](https://modelcontextprotocol.io/docs/2026-07-28/tools/inspector) |

## 参考链接

- <https://github.com/modelcontextprotocol/modelcontextprotocol>
- <https://github.com/modelcontextprotocol>
- <https://modelcontextprotocol.io>
- <https://github.com/modelcontextprotocol/inspector>
