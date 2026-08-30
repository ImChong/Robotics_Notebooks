# Archify（tt-a1i/archify）

> 来源归档

- **标题：** Archify — Agent skill for verifiable architecture, workflow, sequence, data-flow, and lifecycle diagrams
- **类型：** repo（Agent Skill + Node.js CLI / 渲染校验器）
- **作者：** tt-a1i（独立维护；Skill 元数据写明 based on [Cocoon-AI/architecture-diagram-generator](https://github.com/Cocoon-AI/architecture-diagram-generator)，MIT v1.0）
- **链接：** https://github.com/tt-a1i/archify
- **克隆：** `https://github.com/tt-a1i/archify.git`
- **项目页：** https://tt-a1i.github.io/archify/ — [`sources/sites/archify-github-io.md`](../sites/archify-github-io.md)
- **许可：** MIT
- **版本（入库时）：** 稳定发布 **v2.15.0**（2026-08-17）；默认分支开发号 **v2.16.0-dev.0**；`SKILL.md` metadata.version = `2.16`
- **入库日期：** 2026-08-30
- **一句话说明：** 编码代理产出 **类型化 JSON IR**，Archify 确定性编译为 **自包含 HTML/SVG**；五种图（architecture / workflow / sequence / dataflow / lifecycle）带校验门、主题与导出，而不是主题化 Mermaid。
- **开源状态：** **已开源** — MIT；仓内含 `archify/bin/archify.mjs` CLI、`schemas/`、`renderers/`、`examples/`、`archify/SKILL.md`、Proof Lab 产物与 `archify.zip`。项目页 Footer / Install 明确链回本仓。
- **沉淀到 wiki：** 是 → [`wiki/entities/archify.md`](../../wiki/entities/archify.md)

## 仓库概况（2026-08-30 GitHub API / README / 项目页）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`tt-a1i/archify`） |
| 默认分支 | `main` |
| 主要语言 | JavaScript（Node CLI；`npx skills add` 安装） |
| Stars / Forks | ≈32.0k / ≈2.0k |
| 描述 | Agent skill for beautiful, verifiable architecture, workflow, sequence, data-flow, and lifecycle diagrams—self-contained HTML with motion and crisp export. |
| Topics | `agent-skills`, `architecture-diagram`, `diagram-as-code`, `mermaid-alternative`, `coding-agents`, `codex`, `opencode` 等 |
| homepage | https://tt-a1i.github.io/archify/ |
| Skill id | `archify`（`archify/SKILL.md`） |
| 安装 | `npx skills add tt-a1i/archify -g` |

## 为何值得保留

- **可校验的系统图，而不是「看起来像架构」的贴图：** 本站 wiki 页内默认用 Mermaid 表达知识结构；组会、PR、README、对外分享常需要 **可打开、可检索、可导出** 的独立 HTML。Archify 把「代理写 JSON → 校验 → 原子交付」写成 Skill 契约。
- **五种图覆盖机器人栈的沟通面：** 训练/部署拓扑（architecture）、CI 与发布门（workflow）、控制/推理时序（sequence）、数据与评测管线（dataflow）、任务/训练作业状态机（lifecycle）。
- **与 draw.io / Manim / graphify 分工清晰：** [drawio-scientific-illustrator](drawio-scientific-illustrator.md) 产出可编辑科研图元；[Manim](../../wiki/entities/manim.md) 产出讲解视频；[graphify](graphify-labs_graphify.md) 产出可查询知识图。Archify 产出 **已校验的演示/审阅工件**。

## README / Skill 要点（归纳）

- **核心回路：** 代理生成 typed JSON IR → `validate --quality showcase --json`（showcase 需 9 项检查全过）→ `deliver` 原子替换 HTML → 可选 `visual-check` / `preview`。
- **五种图类型：** `architecture`（组件/边界）、`workflow`（泳道过程，新源用 schema v2）、`sequence`（调用与返回）、`dataflow`（管线/血缘）、`lifecycle`（状态、重试、终态）。
- **CLI 入口：** `node archify/bin/archify.mjs` 子命令 `doctor` / `demo` / `guide` / `validate` / `preview` / `deliver` / `compare`（Architecture Delta）/ `brands`。
- **安装面：** Cursor / Claude Code / Codex CLI / OpenCode（`npx skills add`）；Raven 用 `archify.zip` 解到 `~/.raven/workspace/skills/archify`；可选 DeepSeek Harness 社区插件 `@tt-a1i/archify-dsh@0.1.0`（非 DeepSeek 官方产品）。
- **交付物：** 自包含 HTML + PNG / JPEG / WebP / 双主题 SVG / WebM；另有 1200×630 Share Card、Route / Reach Card。
- **交互契约：** 只复用已作者的节点与边；focus / upstream-downstream / route / lens / story **不发明拓扑、不声称运行时影响**。可选 Evidence 节点把 `SRC n` 钉到公开 commit 的文件与行号。
- **Architecture Delta：** 比较两份已校验 snapshot，给出 added / removed / changed / moved / rerouted 的机器回执；不推断风险或合并安全性。
- **更新检查：** 可选 GET 固定稳定 manifest，约 72h 间隔；不上传版本、项目、prompt 或设备 ID。`ARCHIFY_UPDATE_CHECK_DISABLED=1` 关闭。
- **明确不在范围内：** 自动解析 Mermaid 并原样渲染、通用自动布局、托管分享、所见即所得编辑器。
- **协议：** MIT。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 实体页（主） | [`wiki/entities/archify.md`](../../wiki/entities/archify.md) |
| 可编辑科研图对照 | [`wiki/entities/drawio-scientific-illustrator.md`](../../wiki/entities/drawio-scientific-illustrator.md) |
| 讲解动画对照 | [`wiki/entities/manim.md`](../../wiki/entities/manim.md) |
| Web 动效对照 | [`wiki/entities/gsap-skills.md`](../../wiki/entities/gsap-skills.md) |
| 自动知识图对照 | [`wiki/entities/graphify.md`](../../wiki/entities/graphify.md) |
| Agent 时代架构取舍 | [`wiki/concepts/agentic-coding-software-fundamentals.md`](../../wiki/concepts/agentic-coding-software-fundamentals.md) |

## 与本站 sources 的其它锚点

- 项目页：[archify-github-io.md](../sites/archify-github-io.md)
- 科研插图 MCP：[drawio-scientific-illustrator.md](drawio-scientific-illustrator.md)
- 自动构图技能：[graphify-labs_graphify.md](graphify-labs_graphify.md)

## 参考链接

- 源码仓库：<https://github.com/tt-a1i/archify>
- 项目页：<https://tt-a1i.github.io/archify/>
- Scenario guide：<https://tt-a1i.github.io/archify/guide.html>
- Proof Lab：<https://tt-a1i.github.io/archify/gallery.html>
- Skill 入口：<https://github.com/tt-a1i/archify/blob/main/archify/SKILL.md>
- Schema 参考：<https://github.com/tt-a1i/archify/blob/main/archify/schemas/README.md>
- 稳定发布：<https://github.com/tt-a1i/archify/releases/tag/v2.15.0>
