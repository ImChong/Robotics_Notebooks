# Skills For Real Engineers（mattpocock/skills）

> 来源归档

- **标题：** Skills For Real Engineers
- **类型：** repo
- **作者：** Matt Pocock（Total TypeScript / aihero.dev）
- **链接：** https://github.com/mattpocock/skills
- **分发：** https://skills.sh/mattpocock/skills（`npx skills@latest add mattpocock/skills`）
- **入库日期：** 2026-05-20
- **一句话说明：** 作者日常用于「真工程」而非 vibe coding 的可组合编码代理技能库：强调对齐（grill）、共享领域语言（`CONTEXT.md` + ADR）、反馈环（TDD / diagnose）与架构卫生；通过 skills.sh 按需安装，首跑 `/setup-matt-pocock-skills` 绑定 issue tracker 与文档布局。
- **为什么值得保留：** 与本站 [Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md) + [schema/ingest](../../schema/ingest-workflow.md) 同属「把工程习惯写成可版本化文件」；与 [obra/superpowers](obra-superpowers.md)（重流程交付）形成 **轻量可改编 vs 强制方法论** 对照；`productivity/caveman` 与 [JuliusBrussee/caveman](caveman.md) 同名但不同上游，值得并列索引。
- **沉淀到 wiki：** 是 → [`wiki/entities/mattpocock-skills.md`](../wiki/entities/mattpocock-skills.md)

## README 要点（归纳）

- **定位：** *Skills For Real Engineers* — 来自作者 `.claude` 目录的日常技能；反对 GSD / BMAD / Spec-Kit 等「包办流程、削弱控制」的重方法论，主张 **小、可改编、可组合**，适配任意模型。
- **安装（30 秒）：** `npx skills@latest add mattpocock/skills` → 选择技能与 harness → **必须安装** `/setup-matt-pocock-skills` → 在代理中运行该 setup（配置 GitHub/Linear/本地 issue、triage 标签、`CONTEXT.md` 与 `docs/adr/` 路径）。
- **四大失败模式与对应技能（README 叙事）：**
  1. **没做对事** → `/grill-me`、`/grill-with-docs`（对齐 + 共建 ubiquitous language / ADR）
  2. **太啰嗦** → `CONTEXT.md` 共享词汇（`grill-with-docs` 内建）
  3. **代码不对** → `/tdd`（RED-GREEN-REFACTOR）、`/diagnose`（复现→最小化→假设→插桩→修复→回归）
  4. **泥球架构** → `/to-prd`、`/zoom-out`、`/improve-codebase-architecture`（模块边界与 deepening）
- **技能目录（2026-05-20 快照，不含 deprecated/in-progress）：**
  - **engineering：** diagnose, grill-with-docs, improve-codebase-architecture, prototype, setup-matt-pocock-skills, tdd, to-issues, to-prd, triage, zoom-out
  - **productivity：** caveman（极简沟通 ~75% token）、grill-me, handoff, write-a-skill
  - **misc：** git-guardrails-claude-code, migrate-to-shoehorn, scaffold-exercises, setup-pre-commit
- **仓库结构：** 根级 `CONTEXT.md`、`CLAUDE.md`、`docs/`、`skills/`；含 `.claude-plugin` 与示例脚本。
- **协议：** 见仓库 `LICENSE`（README 未单列，以仓库为准）。

## 与本站 sources 的其它锚点

- 流程方法论对照：[obra-superpowers.md](obra-superpowers.md)
- 独立 caveman 压缩技能：[caveman.md](caveman.md)（JuliusBrussee/caveman，非本仓库 `productivity/caveman`）
