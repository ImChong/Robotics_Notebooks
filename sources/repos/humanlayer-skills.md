# HumanLayer Skills（humanlayer/skills）

> 来源归档

- **标题：** HumanLayer Skills
- **类型：** repo
- **作者：** HumanLayer（humanlayer.dev）
- **链接：** https://github.com/humanlayer/skills
- **分发：** `npx skills add humanlayer/skills --skill <SKILLNAME>`（Claude Code marketplace + skills CLI）
- **代码：** https://github.com/humanlayer/skills（**已开源**，MIT）
- **入库日期：** 2026-09-11
- **一句话说明：** HumanLayer 公开的 Claude Code 技能插件集：用 `<important if>` 提升 harness 指令遵从、React 类型收窄、可视化讲解，以及把「传感器–控制器–执行器」控制论隐喻落成可本地运行 + GitHub Actions 调度的迭代代理环。
- **为什么值得保留：** 与本站 [Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md) + [schema/ingest](../../schema/ingest-workflow.md) 同属「把维护流程写成可版本化文件」；`design-control-loop` 把控制论语言显式映射到 **agentic maintenance**，对机器人读者有直觉迁移价值；`build-iterated-agentic-loop` 是 `narrow-react-prop-types` 的通用脚手架，与本仓库 Cloud Agent / `make ci-preflight` 文化同构。
- **沉淀到 wiki：** 是 → [`wiki/entities/humanlayer-skills.md`](../wiki/entities/humanlayer-skills.md)

## README 要点（归纳）

- **定位：** Claude Code skills from HumanLayer；每个 skill 以独立 `plugins/<name>/` 插件发布，含 `.claude-plugin/plugin.json` 与 `skills/<name>/SKILL.md`。
- **安装：** `npx skills add humanlayer/skills --skill SKILLNAME`，再在项目中以 `/skill-name` 调用。
- **技能清单（2026-09-11 快照）：**
  1. **`improve-claude-md`** — 用 `<important if="condition">` 包裹条件相关段落，对抗 Claude Code 对 CLAUDE.md「可能不相关」的系统提醒导致的指令忽略。
  2. **`narrow-react-prop-types`** — 将 React 组件 prop 类型收窄到真实代码路径（排除 Storybook / mock-only 状态）；附带参考型 iterated agentic loop（GHA + agent-memory + PR 标签限流）。
  3. **`build-iterated-agentic-loop`** — 通用脚手架：为目标任务生成 repo-local `SKILL.md`、`.github/workflows/agent-*.yml`、`.github/agent-memory/*.md` 与 references；支持 Claude Code / Codex / OpenCode / CodeLayer 等 headless agent。
  4. **`design-control-loop`** — 访谈式设计 **agentic control loop**：set point（目标状态）→ sensor（测 gap）→ controller（选下一小步）→ actuator（coding agent + skill）→ disturbances + 可选 dampener（回归门）；强调各组件须 **本地可独立运行** 后再接 CI。
  5. **`show-me`** — 用伪代码、调用树、组件树、浅层文件树、Mermaid 或聚焦 HTML artifact 可视化当前话题，少废话。
- **仓库结构：** 根级 `plugins/`（五插件）+ `.claude-plugin/marketplace.json`；各插件 `references/` 含 workflow 模板、agent-runner 模板、`agent-iteration.ts` 等。
- **协议：** MIT（`LICENSE`）。

## 与本站 sources 的其它锚点

- 流程方法论对照：[obra-superpowers.md](obra-superpowers.md)、[mattpocock-skills.md](mattpocock-skills.md)
- 本仓库 harness 规约：[AGENTS.md](../../AGENTS.md)（与 `improve-claude-md` 优化的 CLAUDE.md 同构）
