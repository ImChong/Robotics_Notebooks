# Birdview（Qiuner/birdview）

> 来源归档

- **标题：** Birdview — Map the architecture before every AI code change
- **类型：** repo（Agent Skill + Node.js 校验/渲染器）
- **作者：** Qiuner
- **链接：** https://github.com/Qiuner/birdview
- **克隆：** `https://github.com/Qiuner/birdview.git`
- **项目页：** https://qiuner.github.io/birdview/ — [`sources/sites/birdview.md`](../sites/birdview.md)
- **许可：** MIT
- **入库日期：** 2026-09-21
- **一句话说明：** 面向编码代理的 **Architecture-first** Skill：改代码前先读仓库、产出 `.birdview/architecture.json` 与可选 `activity.jsonl`，经校验渲染为 **自包含 HTML 架构图**，在同一视图上标注 **计划变更范围、源码证据与代理声明的验证结果**。
- **开源状态：** **已开源** — MIT；仓内含 `SKILL.md`、`schemas/`、`scripts/validate.mjs` / `render.mjs` / `birdview.mjs`、`assets/` 查看器模板、`examples/` 虚构 demo 与 `test/` 契约测试。项目页 Install 区链回本仓；npm 包当前标记 private、未发布到 npm registry，安装走 `npx skills add Qiuner/birdview`。
- **沉淀到 wiki：** 是 → [`wiki/entities/birdview.md`](../../wiki/entities/birdview.md)

## 仓库概况（2026-09-21 GitHub API / README / 项目页）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`Qiuner/birdview`） |
| 默认分支 | `main` |
| 主要语言 | TypeScript / JavaScript（Node ≥ 18） |
| Stars / Forks | ≈483 / ≈43 |
| 描述 | Stop letting AI code blind. Map the architecture before every change with Birdview. |
| Topics | `agent-tools`, `architecture-as-code`, `code-visualization`, `coding-agents`, `developer-tools`, `diagram-as-code`, `software-architecture`, `ai-workflow` |
| homepage | https://qiuner.github.io/birdview/ |
| Skill id | `birdview`（仓库根 `SKILL.md`） |
| 安装 | `npx skills add Qiuner/birdview --skill birdview` |

## 为何值得保留

- **补「改之前」这一环：** 日志与 diff 回答「做了什么 / 改了哪些行」，不直接回答「这次改动在系统里占哪块、还会波及谁」。Birdview 把代理对 **模块边界、文件归属、关系与源码证据** 的声明钉在同一张可浏览的图上，再对照计划变更范围。
- **与本站维护者工作流同构：** Robotics_Notebooks 由 Cloud Agent 维护多目录 wiki；在 Isaac Lab / ROS2 / 仿真栈等多仓并行时，**先显式 map 再改** 与 [Superpowers](../../wiki/entities/superpowers-obra.md) 的设计评审、[Open Code Review](../../wiki/entities/open-code-review.md) 的 diff 后评审形成 **前—中—后** 三角。
- **与 Archify 分工清晰：** [Archify](archify.md) 从自然语言或描述生成 **类型化展示图**；Birdview 从 **真实仓库** 归纳 **architecture.json** 并跟踪 **activity.jsonl** 任务活动 — 前者偏对外沟通工件，后者偏 **改码前的范围门禁**。

## README / Skill 要点（归纳）

- **两阶段工作流：** (1) **Map a project** — 代理读源码，创建/更新架构图，模块链到源码证据；(2) **Show changes** — 在同一图上标注计划变更模块/文件、当前步骤与验证记录。
- **数据契约：**
  - `architecture.json` — 项目身份、模块、职责、文件归属、证据、关系、布局；
  - `activity.jsonl`（可选）— 代理声明的任务范围、目标、阶段、验证结果（每行一事件）；
  - `architecture.html` / `activity.html` — 校验通过后生成的自包含查看器（深浅主题、中英切换、关系过滤）。
- **CLI：** `node scripts/validate.mjs`；`node scripts/render.mjs`；`node scripts/birdview.mjs mode auto|on-demand|off` 写项目 `AGENTS.md` / `CLAUDE.md` 模式块。
- **默认 on-demand：** 普通小改不自动触发；可选 **auto** 在每次改码前强制 map。显式调用后 **先展示图与计划范围，等人确认再改码**（agent 引导，非 HTML 写锁）。
- **宿主：** Codex（`/skills`、`$birdview`）、Claude Code（`/birdview`）、DeepSeek Harness 等；slash 命令依宿主而定。
- **明确边界（v0.1）：** 不自动观测代理操作；需重新生成 HTML 并刷新浏览器；无 live transport；`completed` 事件不证明检查通过 — 以 activity 中记录的 check 结果为准；校验不证明架构声明为真或引用文件存在。
- **开发：** `npm ci` / `npm test` / `npm run validate:examples` / `npm run build:demo`；可选 Playwright 浏览器级测试。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 实体页（主） | [`wiki/entities/birdview.md`](../../wiki/entities/birdview.md) |
| 改前架构图 vs 改后评审 | [`wiki/entities/open-code-review.md`](../../wiki/entities/open-code-review.md) |
| 交付流程 skills | [`wiki/entities/superpowers-obra.md`](../../wiki/entities/superpowers-obra.md) |
| 可校验系统图（描述驱动） | [`wiki/entities/archify.md`](../../wiki/entities/archify.md) |
| 中文 Agent 教程生态 | [`wiki/entities/learnprompt.md`](../../wiki/entities/learnprompt.md) |
| 人该保留的取舍判断 | [`wiki/concepts/agentic-coding-software-fundamentals.md`](../../wiki/concepts/agentic-coding-software-fundamentals.md) |

## 与本站 sources 的其它锚点

- 项目页：[birdview.md](../sites/birdview.md)
- 架构图 Skill 对照：[archify.md](archify.md)
- 代码评审 CLI：[open-code-review.md](open-code-review.md)

## 参考链接

- 源码仓库：<https://github.com/Qiuner/birdview>
- 项目页：<https://qiuner.github.io/birdview/>
- Skill 安装：<https://github.com/Qiuner/birdview#quick-start>
- 数据契约：<https://github.com/Qiuner/birdview/blob/main/references/contract.md>
- Stage 1 / 2 工作流：<https://github.com/Qiuner/birdview/blob/main/references/map-project.md>、<https://github.com/Qiuner/birdview/blob/main/references/show-changes.md>
