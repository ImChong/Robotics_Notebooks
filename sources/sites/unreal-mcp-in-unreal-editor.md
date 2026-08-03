# Unreal MCP in Unreal Editor（Epic 官方文档）

> 来源归档

- **标题：** Unreal MCP in Unreal Editor
- **类型：** site（官方技术文档 / Experimental 功能指南）
- **来源：** Epic Games, Inc.（史诗游戏）
- **链接：** https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor
- **文档版本上下文：** Unreal Engine **5.8** Documentation
- **入库日期：** 2026-08-03
- **一句话说明：** Epic 在 **UE 编辑器进程内嵌 MCP server**（插件标识 `ModelContextProtocol`，友好名 **Unreal MCP**），使 Claude Code / Cursor / MCP Inspector 等兼容客户端经本机 HTTP 驱动场景、光照、材质、Slate 检查与自动化测试等编辑器能力；工具本体由 **All Toolsets / Toolset Registry** 提供，而非 MCP 插件自身实现。
- **开源状态：** **部分开源 / 引擎许可边界**
  - **引擎内插件**（`ModelContextProtocol*`、`ToolsetRegistry`、`AllToolsets` 等）随 **Unreal Engine** 分发；完整源码在私有 [EpicGames/UnrealEngine](https://github.com/EpicGames/UnrealEngine)（需 Epic 源码许可），**不是** 独立公开 MCP 仓库。
  - **Claude Code 配套技能插件** **已开源（MIT）**：[`EpicGames/unreal-engine-skills-for-claude-code-plugin`](https://github.com/EpicGames/unreal-engine-skills-for-claude-code-plugin)（见 [repos 归档](../repos/unreal-engine-skills-for-claude-code-plugin.md)）。
- **沉淀到 wiki：** [Unreal MCP](../../wiki/entities/unreal-mcp.md)

---

## 抓取说明

- 以 **2026-08-03** 对文档页公开 HTML 正文抽取为准（Epic Developer Community；`application_version=5.8`）。
- 文档标记为 **Experimental**；API 与数据格式可能随时变化。
- 引擎源码树内路径以文档引用为准（如 `Engine/Plugins/Experimental/ToolsetRegistry/...`）；未获源码许可时无法独立核对私有仓。

---

## 一句话（文档摘要）

**Unreal MCP** 把 MCP server **嵌入 Unreal Editor 进程**，经本机 **Streamable HTTP**（默认 `http://127.0.0.1:8000/mcp`）向任意 MCP 兼容 AI agent 暴露引擎功能；默认仅本机、无认证层，且 Tool 调用在 **game thread 串行**执行。

---

## 为什么值得保留

- 补齐 [UE 5.8 Release Notes](./unreal-engine-5-8-docs.md) 中 **MCP Server (Experimental)** 条目的 **可操作工程细节**（启用、配置、客户端生成、Toolset 扩展、限制）。
- 与本库已有 [FreeCAD MCP](../../wiki/entities/freecad-mcp.md) / [Draw.io Scientific Illustrator](../../wiki/entities/drawio-scientific-illustrator.md) 形成对照：**官方引擎内嵌 MCP** vs **第三方桥接桌面软件**。
- 机器人仿真栈若以 UE 作视觉/场景宿主（AirSim / CARLA / SPEAR / MetaHuman），agentic 场景搭建与资产巡检可直接挂到编辑器，而不另起 Python 反射层。

---

## 核心机制（文档要点）

| 主题 | 要点 |
|------|------|
| **标识** | 源码 / `.uplugin` / C++ / 控制台：`ModelContextProtocol`；Plugin Browser 友好名：**Unreal MCP**；`serverInfo.name`：**`unreal-mcp`** |
| **协议** | [Model Context Protocol](https://modelcontextprotocol.io)：JSON-RPC（`initialize`、`tools/list`、`tools/call` 等）；Primitives：Tools / Resources / Prompts |
| **传输** | **仅 HTTP + SSE**；**不支持** stdio / WebSocket |
| **默认绑定** | `http://127.0.0.1:8000/mcp`；可改端口与 URL path；按 `[HTTPServer.Listeners] DefaultBindAddress`（默认 localhost）绑定，拒绝非 loopback Origin |
| **工具来源** | MCP 插件本身**不实现**业务 Tool；须启用 **All Toolsets**（或按需单个 toolset）+ 依赖 **Toolset Registry** |
| **线程模型** | Tool 在 **game thread 串行**执行；客户端不应重叠发出 Tool 调用 |
| **Tool Search（默认开）** | `tools/list` 只返回三个元工具：`list_toolsets` / `describe_toolset` / `call_tool`，按需发现数百 Tools |

### 能力示例（文档列举）

生成 Actor、配置光照、创建材质实例、检查 Slate widget、运行 automation tests；可扩展自定义 Tools。

### 模块拆分

| 模块 | 角色 |
|------|------|
| `ModelContextProtocol` + `ModelContextProtocolEngine` | **Runtime**：server、协议、设置、控制台命令 |
| `ModelContextProtocolEditor` | **Editor-only**：auto-start；把 Toolset Registry 发现的 toolset 适配为 MCP Tools |

文档明确：Cooked / Shipping 构建也可调用 `IModelContextProtocolModule::StartServer()` 托管 MCP；但 **Registry 适配器是 editor-only**，运行时需 `AddTool()` 显式注册。

---

## 启用与客户端配置（Setup）

1. **Plugins：** 启用 **Unreal MCP** + **All Toolsets**（自动拉起 Toolset Registry）→ 重启编辑器。
2. **Auto Start：** `Edit > Editor Preferences > General > Model Context Protocol` → **Auto Start Server**；或控制台 `ModelContextProtocol.StartServer [port]`。
3. **GenerateClientConfig：**  
   `ModelContextProtocol.GenerateClientConfig <ClaudeCode|Cursor|VSCode|Gemini|Codex|All>`  
   - JSON 客户端（Claude Code / Cursor / VS Code / Gemini）：写入/合并项目根 `.mcp.json`。  
   - Codex：TOML **write-once**（已存在则拒绝覆盖）。
4. 从生成配置所在的 **项目 / workspace 根** 启动 AI agent。
5. **可选：** 启用编辑器内 **Terminal** 插件 + Startup Commands（`TERM=xterm-256color`、`cd` 到 `.mcp.json` 目录、启动 `claude` 等），整条工作流留在编辑器内。

示例 `.mcp.json`（Claude Code）：

```json
{
  "mcpServers": {
    "unreal-mcp": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Quick Start 探针问题示例：`"What actors do I have selected?"` / `"What are a few things you can do in Unreal?"`

---

## Toolset 与扩展（Authoring）

- **Toolset：** 派生自 `UToolsetDefinition` / `unreal.ToolsetDefinition`，经 Registry 在启动时收集，再包装为 MCP Tool。
- **推荐路径：** Python（多数官方 toolset，如 `SceneTools`、`ActorTools`、`MaterialInstanceTools`、`ObjectTools`）或 C++（`UFUNCTION(meta = (AICallable))`）。
- **Claude Code：** 可用 `unreal-mcp` 插件的 **`create-toolset` skill** 脚手架（见 [repos 归档](../repos/unreal-engine-skills-for-claude-code-plugin.md)）。
- **热更新：** `ModelContextProtocol.RefreshTools`；新增 C++ `UFUNCTION` 需完整重启编辑器（Live Coding 不传播新声明）。
- **高级：** 实现 `IModelContextProtocolTool` + `IModelContextProtocolModule::AddTool()` 做动态/运行时 schema。

### 配置与调试摘要

| 类别 | 条目 |
|------|------|
| Editor Preferences | Auto Start（默认 false）、Port 8000、URL Path `/mcp`、Enable Tool Search（默认 true） |
| 控制台命令 | `StartServer` / `StopServer` / `RefreshTools` / `GenerateClientConfig` |
| CLI Flags | `-ModelContextProtocolStartServer`、`-ModelContextProtocolPort=N` |
| 调试 | Output Log 绑定信息；`LogModelContextProtocol`；`npx @modelcontextprotocol/inspector` → Streamable HTTP → `http://127.0.0.1:8000/mcp` |

---

## 限制与已知问题（文档原文要点）

- 仅 HTTP/SSE；无 stdio / WebSocket。
- 默认 loopback；**无认证**；不可安全暴露到本机以外。
- 官方 shipping toolset **不广告** MCP Resources / Prompts。
- Toolset Registry 适配 **仅编辑器**；Cooked 构建须显式 `AddTool()`。
- Live Coding **不**传播新的 `UFUNCTION` 声明。

---

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| Unreal MCP 机制与工程实践 | `wiki/entities/unreal-mcp.md` |
| UE5 / 5.8 引擎宿主总览 | `wiki/entities/unreal-engine-5.md` |
| 对照：桌面 CAD MCP | `wiki/entities/freecad-mcp.md` |
| 对照：桌面 draw.io MCP | `wiki/entities/drawio-scientific-illustrator.md` |

## 参考链接

- 本文档：<https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor>
- MCP 规范：<https://modelcontextprotocol.io>
- MCP Inspector：`npx @modelcontextprotocol/inspector`
- Claude Code 技能插件：<https://github.com/EpicGames/unreal-engine-skills-for-claude-code-plugin>
- UE 5.8 文档总索引归档：[unreal-engine-5-8-docs.md](./unreal-engine-5-8-docs.md)
