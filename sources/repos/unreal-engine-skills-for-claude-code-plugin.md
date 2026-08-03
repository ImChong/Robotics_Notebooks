# Unreal Engine Skills for Claude Code（EpicGames）

- **类型：** repo（Claude Code plugin = Skills + Hooks；依赖编辑器内 Unreal MCP）
- **标题：** Unreal Engine Skills for Claude Code
- **主体：** Epic Games, Inc.
- **链接：** https://github.com/EpicGames/unreal-engine-skills-for-claude-code-plugin
- **许可：** MIT
- **入库日期：** 2026-08-03
- **一句话说明：** 面向 **Claude Code** 的官方插件：提供 **`unreal-mcp` Skill**（驱动 Unreal Editor via MCP 的工作流说明）与 **SessionStart Hook**（注入「本仓库是 UE 项目」上下文）；**不内嵌**静态 `.mcp.json`，须在编辑器中跑 `ModelContextProtocol.GenerateClientConfig ClaudeCode`。
- **开源状态：** **已开源** — GitHub 公开，MIT；Marketplace：`claude-plugins-official` → `unreal-engine-skills-for-claude-code`。
- **关联文档：** [Unreal MCP in Unreal Editor](../sites/unreal-mcp-in-unreal-editor.md)
- **沉淀到 wiki：** [Unreal MCP](../../wiki/entities/unreal-mcp.md)

---

## 仓库元数据（2026-08-03）

| 字段 | 值 |
|------|-----|
| 描述 | Control Unreal Editor directly from Claude Code via MCP |
| Stars（约） | 169 |
| 默认分支 | `main` |
| 安装（单开发者） | `/plugin install unreal-engine-skills-for-claude-code@claude-plugins-official` |

## 内容结构（README）

| 组件 | 路径 / 角色 |
|------|-------------|
| Skill | `skills/unreal-mcp` — 经 MCP 驱动编辑器的说明与工作流；含 `create-toolset` 脚手架约定 |
| Hook | `hooks/unreal-context.sh` — SessionStart（startup / resume / clear）注入 UE 约定提示 |
| 平台 | macOS / Linux；Windows 需 **Git Bash 或 WSL**（原生 PowerShell 无 bash 时 hook 失效，MCP 工具仍可用） |

## 前置条件（与官方 MCP 文档一致）

1. Unreal Editor 启用 **ModelContextProtocol** + **AllToolsets**
2. MCP server 已启动（`ModelContextProtocol.StartServer` 或 Auto Start）
3. 项目根已生成 `.mcp.json`（`GenerateClientConfig ClaudeCode`）
4. Claude Code `/mcp` 可见已连接的 `unreal-mcp`

## 安全提示（README）

- 插件赋予 Claude **对运行中编辑器的宽泛实时访问**；等同于助手执行任意本机代码。
- **localhost 不是信任边界**：同用户同机任意进程可连；Origin 校验只挡浏览器页，勿在共享/不可信机器上开 server，勿把端口暴露到 loopback 外。

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| Unreal MCP 客户端与技能层 | `wiki/entities/unreal-mcp.md` |
| Epic 公开仓索引 | `sources/repos/epicgames-github-org.md` |

## 参考链接

- <https://github.com/EpicGames/unreal-engine-skills-for-claude-code-plugin>
- <https://dev.epicgames.com/documentation/unreal-engine/unreal-mcp-in-unreal-editor>
- <https://claude.com/plugins>
