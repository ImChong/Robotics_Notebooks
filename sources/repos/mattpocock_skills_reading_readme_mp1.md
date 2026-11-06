# mattpocock/skills README 通读

> 来源归档（原文通读 · 支撑 `wiki/entities/mattpocock-skills.md`）

- **标题：** Skills For Real Engineers（上游仓库 README）
- **作者：** Matt Pocock
- **类型：** repo / agent-skills
- **原文：** <https://github.com/mattpocock/skills>（2026-11 经 raw.githubusercontent.com 获取 `README.md` 全文，258 行）
- **一句话说明：** 作者日常用的可组合编码代理技能库；README 以四类失败模式（没对齐、太啰嗦、代码不对、泥球架构）组织技能，并给出各 harness 的安装方式与 Reference 目录。
- **核对范围：** 通读 README 全文（定位、安装、四类失败模式、Engineering / Productivity 技能 Reference）；未逐个阅读各 `skills/**/SKILL.md` 正文、`CONTEXT.md`、`docs/`，子技能细节以对应 wiki 子页与 skills.sh 归档为准。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位：** 「真工程而非 vibe coding」；GSD、BMAD、Spec-Kit 等方案接管流程，会削弱控制并使流程中的 bug 难排查；本库技能小、易改、可组合，适配任意模型。
- **安装：** Claude Code 用 `claude plugin install mattpocock-skills@claude-plugins-official`（自动更新）；Codex、Copilot 有各自 marketplace 命令；Gemini CLI 手动装 `skills/engineering` 与 `skills/productivity`；其它 agent 用 `npx skills@latest add mattpocock/skills`（需包含 `setup-matt-pocock-skills`，`npx skills@latest update` 更新）。插件与 skills.sh 二选一，否则每个技能装两遍。
- **一次性 setup：** 每仓库运行 `/setup-matt-pocock-skills`：选 issue tracker（GitHub / GitLab / 本地文件等）、triage 用的标签、文档存放位置。
- **失败模式 1（没做对事）：** 引 Pragmatic Programmer「没人确切知道自己想要什么」；解法是 grilling，用 `/grill-me`（非代码）与 `/grill-with-docs`。
- **失败模式 2（太啰嗦）：** 引 Eric Evans 的通用语言；用共享术语文档让代理少用「20 个词代替 1 个词」；README 示例「materialization cascade」；好处还含命名一致、代码库更易导航、思考 token 更少。
- **失败模式 3（代码不对）：** 需要静态类型、浏览器访问、自动化测试等反馈环；`/tdd` 做 red-green-refactor；`/diagnosing-bugs` 为分阶段门控的调试环。
- **失败模式 4（泥球架构）：** 引 Kent Beck、Ousterhout「深模块」；`/to-spec` 先问清触及哪些模块；`/improve-codebase-architecture` 巡检「加深」机会，建议每隔几天跑一次，它是巡检而非拯救。
- **Reference 分类：** 按「谁可调用」分用户调用（编排）与模型调用（可复用纪律）；用户调用技能不得再调用另一个用户调用技能。Engineering 用户调用：ask-matt、grill-with-docs、triage、improve-codebase-architecture、setup-matt-pocock-skills、to-spec、to-tickets、implement、implement-spec、wayfinder、retro；模型调用：prototype、diagnosing-bugs、research、tdd、domain-modeling、codebase-design、code-review、pr、wizard。Productivity 用户调用：grill-me、handoff、teach、to-questionnaire、wait-what；模型调用：grilling、writing-for-agents。
- **与既有归档的差异：** 本次 README 中 `GLOSSARY.md` 取代了先前常见的 `CONTEXT.md` 叫法（README 注明示例仓库固定提交仍用旧名），`/diagnose`、`to-prd`、`to-issues` 已见为 `diagnosing-bugs`、`to-spec`、`to-tickets`，并新增 implement、wayfinder、retro 等；README 未再列 caveman、zoom-out 与 misc 目录（是否仍存在于仓库未核对）。

## 对 wiki 的映射

- [`wiki/entities/mattpocock-skills.md`](../../wiki/entities/mattpocock-skills.md)
