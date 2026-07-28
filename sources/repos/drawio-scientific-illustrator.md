# drawio-scientific-illustrator（icebird1998/drawio-scientific-illustrator）

> 来源归档

- **标题：** Draw.io Scientific Illustrator — Live MCP control of the visible draw.io canvas
- **类型：** repo（Codex plugin = MCP servers + Agent Skill + marketplace 元数据）
- **作者：** icebird1998
- **链接：** https://github.com/icebird1998/drawio-scientific-illustrator
- **克隆：** `https://github.com/icebird1998/drawio-scientific-illustrator.git`
- **许可：** MIT
- **版本（入库时）：** v1.0.0（2026-07-12 首发）
- **入库日期：** 2026-07-28
- **一句话说明：** 让 **Codex** 经本机 **MCP** 直接调用桌面版 **draw.io** 的图模型 API，在**可见画布**上逐步绘制可编辑科研插图；不是 OS 键鼠自动化，也不是「先写 XML 再打开」。
- **开源状态：** **已开源** — MIT；含 `drawio-live` / `drawio-file-utils` MCP、`recreate-scientific-figure-in-drawio` Skill、安装脚本与 CI smoke test。无独立项目页（`homepage` 为空，以 GitHub README 为准）。
- **沉淀到 wiki：** 是 → [`wiki/entities/drawio-scientific-illustrator.md`](../../wiki/entities/drawio-scientific-illustrator.md)

## 仓库概况（2026-07-28 GitHub API / README）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`icebird1998/drawio-scientific-illustrator`） |
| 默认分支 | `main` |
| 主要语言 | JavaScript（Node.js ≥22） |
| Stars / Forks | ~1.0k / ~71 |
| 描述 | Live MCP control of the visible draw.io canvas for step-by-step scientific illustration in Codex. |
| Topics | `codex`, `diagrams`, `drawio`, `mcp`, `scientific-figures`, `scientific-illustration` |
| 插件 id | `drawio-scientific-illustrator@drawio-scientific-tools` |
| Skill | `recreate-scientific-figure-in-drawio` |

## 为何值得保留

- **科研插图 Agent 闭环：** 本站 wiki / 论文解读常需 **可编辑框图、管线示意、对比图**；本仓库把「看参考图 → 分解图元 → 可见逐步绘制 → 截图审图 → 保存/导出」写成 **Skill + MCP**，对齐本库维护者「代理画图而非黑盒贴图」的需求。
- **MCP 操控桌面专业软件样本：** 与 [FreeCAD MCP](freecad-mcp.md) 同属 **本机专业 GUI + localhost MCP**；本仓垂直在 **draw.io 矢量科研图**，强调 **可见步进** 与 **拒绝 XML-first / OS 自动化**。
- **与 Manim / Mermaid 分工清晰：** [Manim](../../wiki/entities/manim.md) 偏 **程序化讲解动画**；本库页内 Mermaid 偏 **知识图结构**；本工具偏 **可编辑 `.drawio` 交付物**（PNG/SVG/PDF 导出）。

## README / Skill 要点（归纳）

- **双 MCP：**
  - `drawio-live`（`scripts/live-server.mjs`）：启动/连接可见 draw.io，实时改 graph。
  - `drawio-file-utils`（`scripts/server.mjs`）：校验已保存 `.drawio`，导出 PNG/SVG/PDF/JPG。
- **Live 工具（Skill 主路径）：** `drawio_live_launch` → `status` → `add_shape` / `add_edge` / `draw_sequence` → `screenshot` → `inspect` / `update_cell` → `fit` → `save_snapshot`；另有 `drawio_live_clear`。
- **文件工具（Skill 要求保存后才用）：** `drawio_validate`、`drawio_export` 等；文件侧另有 `drawio_create_diagram` / `write_xml` 等，但 Skill **硬边界禁止 XML-first 作为主绘制法**。
- **硬边界：** 只控 draw.io 内部 graph API；禁止 OS 键鼠/窗口自动化；禁止先生成 XML 再打开；截图仅用于审 **draw.io 渲染区**。
- **依赖：** Codex（desktop/CLI 插件支持）+ 本机 [draw.io desktop](https://www.drawio.com/) + Git；脱离 Codex 运行 MCP 时需 Node.js ≥22。可选 `DRAWIO_PATH` / `DRAWIO_LIVE_PORT`（默认偏好 `9333`）/ `DRAWIO_LIVE_PROFILE`。
- **安装：** `codex plugin marketplace add <repo>` + `codex plugin add drawio-scientific-illustrator@drawio-scientific-tools`；或 `install.sh` / `install.ps1`；安装后需重启 Codex 并**新建任务**。
- **安全：** 调试端口仅绑 `127.0.0.1`；只附着可识别为 draw.io/diagrams.net 的页面；无遥测（见 `PRIVACY.md`）。
- **平台：** v1.0.0 **Windows 已测**；macOS/Linux 尽力支持。
- **局限：** 显微图/热图/复杂数据图等需栅格插入能力（尚未有专用 live image 工具）；还原度受参考图分辨率与「能否用 draw.io 图元表达」约束。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 实体页（主） | [`wiki/entities/drawio-scientific-illustrator.md`](../../wiki/entities/drawio-scientific-illustrator.md) |
| 桌面 CAD MCP 对照 | [`wiki/entities/freecad-mcp.md`](../../wiki/entities/freecad-mcp.md) |
| Agent Skills 对照 | [`wiki/entities/cad-skills.md`](../../wiki/entities/cad-skills.md)、[`wiki/entities/img2threejs.md`](../../wiki/entities/img2threejs.md)、[`wiki/entities/gsap-skills.md`](../../wiki/entities/gsap-skills.md) |
| 讲解动画对照 | [`wiki/entities/manim.md`](../../wiki/entities/manim.md) |

## 参考链接

- 源码仓库：<https://github.com/icebird1998/drawio-scientific-illustrator>
- Skill 入口：`plugins/.../skills/recreate-scientific-figure-in-drawio/SKILL.md`
- 隐私说明：<https://github.com/icebird1998/drawio-scientific-illustrator/blob/main/PRIVACY.md>
- draw.io 桌面版：<https://www.drawio.com/>
- MCP 规范：<https://modelcontextprotocol.io>
