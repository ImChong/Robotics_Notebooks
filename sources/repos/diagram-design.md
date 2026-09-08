# Diagram Design（cathrynlavery/diagram-design）

> 来源归档

- **标题：** Diagram Design — Editorial diagrams your designer won't hate
- **类型：** repo（Agent Skill + 多 harness marketplace / plugin 元数据）
- **作者：** Cathryn Lavery（[littlemight.com](https://littlemight.com)、[BestSelf.co](https://bestself.co)）
- **链接：** https://github.com/cathrynlavery/diagram-design
- **项目页：** https://cathrynlavery.github.io/diagram-design/
- **克隆：** `https://github.com/cathrynlavery/diagram-design.git`
- **许可：** MIT
- **版本（入库时）：** v2.5.10（README / gallery；GitHub API `updated_at` 2026-09-08）
- **入库日期：** 2026-09-08
- **一句话说明：** 39 种 editorial 图表类型的 Agent Skill：自包含 HTML + SVG，无阴影、无「Mermaid slop」；可从网站 onboarding 品牌色，并把 draw.io / Mermaid 源 **重绘** 为同一设计系统下的交付物（HTML / SVG / PNG）。
- **开源状态：** **已开源** — MIT；`skills/diagram-design/` 为共享 Skill 根；本地 gallery `skills/diagram-design/assets/index.html` 可直接浏览器打开，无构建步骤。
- **沉淀到 wiki：** 是 → [`wiki/entities/diagram-design.md`](../../wiki/entities/diagram-design.md)

## 仓库概况（2026-09-08 GitHub API / README）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`cathrynlavery/diagram-design`） |
| 默认分支 | `main` |
| 主要语言 | HTML |
| Stars / Forks | ~33.4k / ~2.1k |
| Topics | `agent-skills`, `claude-code`, `codex`, `data-visualization`, `diagrams`, `drawio`, `mermaid`, `svg` |
| Skill 路径 | `skills/diagram-design/`（`SKILL.md` + `references/` + `assets/`） |
| 插件 id | `diagram-design@diagram-design` |

## 为何值得保留

- **本站 wiki 用 Mermaid 表达知识结构，对外沟通常需 editorial 级静态图。** 本 Skill 把「架构 / 时序 / 数据流 / 象限 / 飞轮 / Sankey / Wardley …」编译成 **可直接截图或导出 PNG/SVG 的自包含 HTML**，与页内 Mermaid 分工清晰。
- **与 Archify / Draw.io Scientific Illustrator 形成三角对照：** [Archify](../../wiki/entities/archify.md) 偏 **类型化 JSON IR + Node 校验** 的系统图；[Draw.io Scientific Illustrator](../../wiki/entities/drawio-scientific-illustrator.md) 偏 **Codex MCP 可见步进 `.drawio`**；本仓偏 **editorial 设计系统 + 39 种版式 + draw.io/Mermaid 重绘**。
- **多 harness 安装面完整：** Claude Code / Codex / Factory Droid / Pi / Kiro / OpenCode 等均有文档路径；含 `/import-drawio`、`/import-mermaid`、`/export-diagram`、品牌 `/profile` 与 `/doctor` 运维模板。
- **语义模式与可选动效：** `semantic-patterns.md` 把队列、策略 trace、信任边界等行为与 **最近邻视觉类型** 解耦；`animation.md` 提供可访问的 `reveal` / `step` / `loop`，默认仍为 **无脚本静态 HTML**。

## README / Skill 要点（归纳）

- **39 种视觉类型**（v2.5.10 末批 ten 型含 Sankey、fishbone、Wardley、kanban、user journey、deployment、dependency、UML class、story map、db schema 等），每种有 **minimal light / minimal dark / full-editorial** 三变体。
- **输出契约：** 自包含 HTML + 内联 SVG；默认 **无 JavaScript、无外部图片依赖**；`role="img"` + `aria-labelledby` 默认可访问。
- **品牌 onboarding：** 抓取站点 CSS → 映射 `paper` / `ink` / `accent` 等语义 token 写入 `references/style-guide.md`；WCAG AA 对比度自动校验；多项目可用 `~/.diagram-design/profiles/` + `.diagram-design` marker。
- **Import 四旋钮：** `format`（html/svg/png）、`size`（slide-16x9、social-og 等）、`detail`（faithful/balanced/simplified）、`audience`（engineer/mixed/executive）；结束时输出 **fidelity ledger**（合并/丢弃节点清单）。
- **Export：** `/export-diagram`（Pi）或 `/diagram-design:export-diagram`（Claude Code）从 HTML 导出 SVG/PNG（可调 scale）。
- **安装示例：**
  - Claude Code：`/plugin marketplace add cathrynlavery/diagram-design` → `/plugin install diagram-design@diagram-design`
  - Codex：`codex plugin marketplace add cathrynlavery/diagram-design` → `codex plugin add diagram-design@diagram-design`
  - Pi：`pi install https://github.com/cathrynlavery/diagram-design`
  - Cursor 可编辑安装：`ln -s ~/code/diagram-design/skills/diagram-design ~/.cursor/skills/diagram-design`
- **设计哲学（README 引语）：** *The highest-quality move is usually deletion.* 目标密度约 4/10；accent 只留给 1–2 个读者应先看的节点。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 实体页（主） | [`wiki/entities/diagram-design.md`](../../wiki/entities/diagram-design.md) |
| 可校验系统图对照 | [`wiki/entities/archify.md`](../../wiki/entities/archify.md) |
| 可见 draw.io 步进对照 | [`wiki/entities/drawio-scientific-illustrator.md`](../../wiki/entities/drawio-scientific-illustrator.md) |
| Agent Skills 生态 | [`wiki/entities/mattpocock-skills.md`](../../wiki/entities/mattpocock-skills.md)、[`wiki/entities/gsap-skills.md`](../../wiki/entities/gsap-skills.md) |
| 项目页归档 | [`sources/sites/diagram-design-cathrynlavery-github-io.md`](../sites/diagram-design-cathrynlavery-github-io.md) |

## 参考链接

- 源码仓库：<https://github.com/cathrynlavery/diagram-design>
- 在线 gallery：<https://cathrynlavery.github.io/diagram-design/>
- Skill 入口：`skills/diagram-design/SKILL.md`
- Cookbook：<https://github.com/cathrynlavery/diagram-design/blob/main/docs/cookbook.md>
