# freecad-mcp

- **标题**：FreeCAD MCP
- **类型**：repo
- **来源**：neka-nat（GitHub）
- **链接**：<https://github.com/neka-nat/freecad-mcp>
- **克隆**：`https://github.com/neka-nat/freecad-mcp.git`
- **PyPI**：`freecad-mcp`（`uvx freecad-mcp`）
- **入库日期**：2026-07-10
- **一句话说明：** 面向 **Claude Desktop / 通用 MCP 宿主** 的 **FreeCAD 远程控制桥**：FreeCAD 内 **MCP Addon 工作台** 启动 **RPC 服务**，外部 Python **`freecad-mcp` MCP server** 经 MCP 暴露建文档、改对象、执行 Python、`get_view` 截图、**CalculiX FEM** 等工具——让编码代理用自然语言驱动已安装的桌面 FreeCAD，而非另起脚本 CAD 栈。
- **沉淀到 wiki：** 是 → [`wiki/entities/freecad-mcp.md`](../../wiki/entities/freecad-mcp.md)

## 仓库概况（2026-07-10 GitHub API / README）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`neka-nat/freecad-mcp`） |
| 默认分支 | `main` |
| 主要语言 | Python |
| Stars / Forks | ~1.3k / ~188 |
| 描述 | FreeCAD MCP(Model Context Protocol) server |
| 许可 | MIT |
| Topics | `claude`, `freecad`, `mcp` |
| Python | `>=3.12`（`pyproject.toml`） |
| 包版本 | 0.1.19（入库时） |

## 架构（README 摘要）

**双组件：**

1. **FreeCAD Addon**（`addon/FreeCADMCP`）— 安装到各平台 `Mod/` 目录；在 FreeCAD 内切换 **「MCP Addon」工作台**，通过工具栏 **Start RPC Server** 启动本地 RPC（支持 **Auto-Start**、**Remote Connections** + **Allowed IPs** 白名单）。
2. **MCP Server**（PyPI `freecad-mcp`）— 由 `uvx freecad-mcp` 或 `uv run freecad-mcp` 启动；在 Claude Desktop 等 MCP 宿主中注册为 `freecad` server；经 RPC 调用 FreeCAD Python API。

**典型宿主配置（Claude Desktop `claude_desktop_config.json`）：**

```json
{
  "mcpServers": {
    "freecad": {
      "command": "uvx",
      "args": ["freecad-mcp"]
    }
  }
}
```

可选 `--only-text-feedback`（省 token，仅文本反馈）、`--host <ip>`（远程 FreeCAD 机器）。

## MCP 工具列表（README）

| 工具 | 作用 |
|------|------|
| `create_document` | 新建 FreeCAD 文档 |
| `create_object` | 创建对象 |
| `edit_object` | 编辑对象 |
| `delete_object` | 删除对象 |
| `execute_code` | 在 FreeCAD 内执行任意 Python |
| `insert_part_from_library` | 从 [FreeCAD-library](https://github.com/FreeCAD/FreeCAD-library) 插入标准件 |
| `get_view` | 活动视图截图（视觉反馈闭环） |
| `get_objects` / `get_object` | 枚举/读取文档对象 |
| `get_parts_list` | 零件库列表 |
| `run_fem_analysis` | 对 `Fem::FemAnalysis` 跑 **CalculiX** 求解并返回摘要（最大 von Mises 应力、最大位移等） |

## 演示场景（README）

- 自然语言设计 **法兰**、**玩具车** 等参数件（含 GIF demo）
- 从 **2D 工程图** 输入建模（附 Claude 对话分享链接）
- FEM 悬臂梁示例：`examples/cantilever_fem.py`

## 与机器人研究/工程的关联点

- **桌面 CAD 代理化：** 与 [CAD Skills](../../wiki/entities/cad-skills.md) 的 **build123d 无头 CLI** 路线互补——本仓库直接操控用户已安装的 [FreeCAD](../../wiki/entities/freecad.md)，保留 **Part Design / Assembly / Robot 工作台** 与社区 **URDF 插件** 生态。
- **视觉闭环：** `get_view` 截图支持代理 **审图迭代**（法兰、支架、夹具草模），对齐 [文字生成 CAD](../../wiki/concepts/text-to-cad.md) 的「人类/代理在环」实践。
- **结构初评：** `run_fem_analysis` 可把 **FEM 工作台 + CalculiX** 纳入代理工作流——适合支架刚度粗评，**不替代** 动力学标定或 RL 仿真（参见 [仿真物理保真度](../../wiki/queries/simulation-physics-fidelity.md)）。
- **MCP 生态样本：** 与 [DimOS](../../wiki/entities/dimensionalos-dimos.md)、[Hermes Agent](../../wiki/entities/hermes-agent.md)、[ppf-contact-solver](../../wiki/entities/ppf-contact-solver.md) 等 **MCP 驱动专业软件** 同类，但垂直在 **机械 CAD**。

## 对 wiki 的映射

- 升格页面：[wiki/entities/freecad-mcp.md](../../wiki/entities/freecad-mcp.md)
- 交叉引用：[wiki/entities/freecad.md](../../wiki/entities/freecad.md)、[wiki/entities/cad-skills.md](../../wiki/entities/cad-skills.md)、[wiki/concepts/text-to-cad.md](../../wiki/concepts/text-to-cad.md)、[wiki/entities/step2urdf.md](../../wiki/entities/step2urdf.md)

## 参考链接

- 源码仓库：<https://github.com/neka-nat/freecad-mcp>
- FreeCAD 官方：<https://www.freecad.org>
- FreeCAD-library：<https://github.com/FreeCAD/FreeCAD-library>
- MCP 规范：<https://modelcontextprotocol.io>
