# Ponytail（DietrichGebert/ponytail）

> 来源归档

- **标题：** Ponytail
- **类型：** repo
- **作者：** Dietrich Gebert
- **链接：** https://github.com/DietrichGebert/ponytail
- **入库日期：** 2026-07-03
- **一句话说明：** 面向 20+ 编码代理 harness 的可安装技能/插件：用「懒但资深」的 **必要性阶梯（YAGNI → 复用 → stdlib → 原生 → 依赖 → 一行 → 最小实现）** 约束代理 **少写代码** 而非少做思考；宣称在真实 agentic 基准上平均约 **-54% LOC、-22% token、-20% cost、-27% time** 且 **100% 安全**（验证/错误处理/安全/无障碍不砍）。
- **为什么值得保留：** 与本站 [Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md) + [schema/ingest](../../schema/ingest-workflow.md) 维护强相关：本仓库 ingest 会反复改脚本与派生文件，ponytail 针对 **过度工程化（装库写 wrapper）** 而非单纯 **输出措辞**；与 [Caveman](caveman.md)（压缩 mouth）、[Superpowers](obra-superpowers.md)（交付流程）形成 **代码量 / 措辞 / 流程** 三角对照；README 含可复现 agentic benchmark（FastAPI+React 模板、12 任务、n=4）。
- **沉淀到 wiki：** 是 → [`wiki/entities/ponytail.md`](../wiki/entities/ponytail.md)

## README 要点（归纳）

- **定位叙事：** 「长马尾资深工程师」— 你给他五十行，他换成一行；ponytail 把该品味写进代理默认行为。
- **核心机制 — 必要性阶梯（写代码前停在上层即可）：**
  1. 这功能需要存在吗？→ YAGNI 跳过
  2. 代码库里已有？→ 复用
  3. 标准库能做？→ 用 stdlib
  4. 平台原生能力？→ 用原生（如 `<input type="date">`）
  5. 已装依赖？→ 用依赖
  6. 一行能写？→ 一行
  7. 最后才写「能工作的最小实现」
- **安全红线：** 信任边界校验、数据丢失处理、安全、无障碍 **永不削减**；与裸「YAGNI + one-liners」提示词对照时后者 safety 约 95%。
- **强度档位：** `lite` / `full`（默认）/ `ultra` / `off`；`PONYTAIL_DEFAULT_MODE` 或 `~/.config/ponytail/config.json`。
- **命令族：** `/ponytail`、`/ponytail-review`（diff 过度工程审查）、`/ponytail-audit`（全仓）、`/ponytail-debt`（延后 shortcut 台账）、`/ponytail-gain`（基准成绩板）、`/ponytail-help`。
- **多 harness：** Claude Code / Codex / Copilot CLI / Pi / OpenCode / Gemini / Antigravity / Hermes / Devin / OpenClaw / Swival / CodeWhale 等；Cursor/Windsurf/Cline/Aider/Kiro/Zed 等 **复制规则文件**（`.cursor/rules/`、`AGENTS.md` 等），见 `docs/agent-portability.md`。
- **生命周期 hook：** Claude Code / Codex 等用 Node.js hook 每轮注入规则；需 `node` 在 PATH。
- **基准（agentic，2026-06-18）：** headless Claude Code 编辑 [full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template)，12 feature tickets，Haiku 4.5，n=4；对照无 skill、caveman、裸 YAGNI prompt。ponytail 唯一 **全指标下降且 100% safe**；date picker 等 over-build 陷阱降幅最大（404→23 行），已极简代码接近 0%。详见 `benchmarks/results/2026-06-18-agentic.md`。
- **协议：** MIT。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 实体页（主） | [`wiki/entities/ponytail.md`](../../wiki/entities/ponytail.md) |
| 输出压缩对照 | [`wiki/entities/caveman.md`](../../wiki/entities/caveman.md) |
| 交付流程对照 | [`wiki/entities/superpowers-obra.md`](../../wiki/entities/superpowers-obra.md) |
| 轻量技能对照 | [`wiki/entities/mattpocock-skills.md`](../../wiki/entities/mattpocock-skills.md) |
| LLM Wiki 范式 | [`wiki/references/llm-wiki-karpathy.md`](../../wiki/references/llm-wiki-karpathy.md) |

## 与本站 sources 的其它锚点

- 输出 token 压缩：[caveman.md](caveman.md)（ponytail 基准对照臂之一）
- 流程方法论：[obra-superpowers.md](obra-superpowers.md)
- 日常工程技能：[mattpocock-skills.md](mattpocock-skills.md)
