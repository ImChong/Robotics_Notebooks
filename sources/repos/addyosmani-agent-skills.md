# Agent Skills（addyosmani/agent-skills）

> 来源归档

- **标题：** Agent Skills
- **类型：** repo
- **作者：** Addy Osmani（[@addyosmani](https://github.com/addyosmani)）；协作者 Federico Bartoli、Joan León
- **链接：** https://github.com/addyosmani/agent-skills
- **官方站：** https://skills.addy.ie
- **分发：** `npx skills add addyosmani/agent-skills`（[vercel-labs/skills](https://github.com/vercel-labs/skills)）；Claude/Codex marketplace 插件；各 harness 见 `docs/*-setup.md`
- **入库日期：** 2026-09-19
- **协议：** MIT
- **一句话说明：** 面向编码代理的 **25 项生产级工程技能** + **9 个生命周期 slash 命令**：把 spec、TDD、五轴 code review、安全/性能/可观测性、CI/CD 与 ship 检查清单写成带 **anti-rationalization** 与 **evidence 出口** 的结构化 `SKILL.md` 工作流；嵌入 Google 工程文化实践。
- **为什么值得保留：** 与 [obra/superpowers](../../wiki/entities/superpowers-obra.md)、[mattpocock/skills](../../wiki/entities/mattpocock-skills.md) 构成 **重流程 / 轻组合 / 全 SDLC 技能包** 三角对照；对本站 Cloud Agent 维护（ingest、`make ci-preflight`、PR 评审）有直接借鉴价值。
- **沉淀到 wiki：** 是 → [`wiki/entities/agent-skills-addyosmani.md`](../wiki/entities/agent-skills-addyosmani.md)

## README 要点（归纳）

- **定位：** *Production-grade engineering skills for AI coding agents* — 技能是 **代理要执行的流程**（步骤、检查点、退出条件），不是被动阅读的参考文档。
- **生命周期（9 commands → 6 phases）：** DEFINE（`/spec`）→ PLAN（`/plan`）→ BUILD（`/build`，可选 **`/build auto`**）→ VERIFY（`/test`）→ REVIEW（`/review`、`/webperf`、`/code-simplify`、`/constraints`）→ SHIP（`/ship`）。
- **25 skills：** 24 生命周期技能 + meta `using-agent-skills`；涵盖 interview-me、spec-driven-development、constraint-driven-development、incremental-implementation、TDD、doubt-driven-development、code-review-and-quality、security-and-hardening、ci-cd-and-automation、shipping-and-launch 等。
- **Agent personas（4）：** code-reviewer、test-engineer、security-auditor、web-performance-auditor；`/ship` 可并行 fan-out。
- **References（7）：** definition-of-done、testing-patterns、security/performance/accessibility/observability checklists、orchestration-patterns。
- **仓库分层：** 便携核心 `skills/` + `agents/` + `references/`；各 harness 适配目录（`.claude/`、`.codex-plugin/`、`commands/` 等）。
- **质量机制：** `evals/` 三层 eval（结构、路由、执行 trace）；CI workflow。
- **已知局限（README）：** 单 skill `npx` 安装 **不复制** 仓库级 `references/`（[#361](https://github.com/addyosmani/agent-skills/issues/361)）；Antigravity CLI 部分 wrapper 命令不可发现（见 `docs/antigravity-setup.md`）。

## 与本站 sources 的其它锚点

- 项目页核查：[`sources/sites/skills-addy-ie.md`](../sites/skills-addy-ie.md)
- 流程对照：[obra-superpowers.md](obra-superpowers.md)、[mattpocock-skills.md](mattpocock-skills.md)
- 评审工具对照：[open-code-review.md](open-code-review.md)
