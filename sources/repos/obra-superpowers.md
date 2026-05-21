# Superpowers（obra/superpowers）

> 来源归档

- **标题：** Superpowers
- **类型：** repo
- **来源：** Jesse Vincent（obra，Prime Radiant）
- **链接：** https://github.com/obra/superpowers
- **关联市场：** https://github.com/obra/superpowers-marketplace（Claude Code / Copilot CLI 等插件分发）
- **入库日期：** 2026-05-17
- **一句话说明：** 面向多种「编码代理 harness」的可安装技能库 + 启动规约：在写代码前强制走头脑风暴→设计确认→可执行计划→子代理实现与评审→TDD 与 git worktree 隔离等流程；技能以 `SKILL.md` 等形式可被代理检索并按规约执行。
- **为什么值得保留：** 与本知识库采用的 **Karpathy LLM Wiki + schema/ingest** 思路相邻：都是把「人类策展 + LLM 执行」写成可重复的流程与文件契约；对维护本仓库的 agent 工作流有对照价值。
- **沉淀到 wiki：** 是 → [`wiki/entities/superpowers-obra.md`](../wiki/entities/superpowers-obra.md)

## README 要点（归纳）

- **定位：** *complete software development methodology for your coding agents*，基于可组合 **skills** 与 **session 启动时的初始指令**，促使代理在开工前澄清目标而非直接写代码。
- **核心叙事：** 设计文档分段展示供人审阅 → 批准后生成足够细的实施计划（强调 **RED/GREEN TDD**、**YAGNI**、**DRY**）→ **subagent-driven development** 或带检查点的批量执行 → 任务间 **code review**；技能在任务前自动匹配，属 **mandatory workflows**。
- **多 harness 安装：** Claude Code（官方或 Superpowers marketplace 插件）、Codex CLI/App、Factory Droid、Gemini CLI、OpenCode、Cursor（`/add-plugin superpowers` 或市场搜索）、GitHub Copilot CLI 等；各环境需**分别安装**。
- **技能分类（README 列举）：** Testing（如 `test-driven-development`）、Debugging（`systematic-debugging`、`verification-before-completion`）、Collaboration（`brainstorming`、`writing-plans`、`executing-plans`、`subagent-driven-development`、`using-git-worktrees` 等）、Meta（`writing-skills`、`using-superpowers`）。
- **哲学：** TDD 优先；系统化优于临时摸索；以简化为目标；先验证再宣称完成。
- **协议：** MIT。
- **贡献说明：** 一般不接受随意新增技能；修改需跨所支持 harness 可用；详见仓库内 `skills/writing-skills/SKILL.md` 与 PR 模板。

## 与本站 sources 的其它锚点

- 发布叙述与背景：[`sources/blogs/fsck_superpowers_announcement_2025-10-09.md`](../blogs/fsck_superpowers_announcement_2025-10-09.md)
