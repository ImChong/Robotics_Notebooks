# claude-api（Anthropic 官方 Agent Skill）

> 来源归档

- **标题：** claude-api（`anthropics/skills` 子目录）
- **类型：** repo / agent-skill
- **作者：** Anthropic（技能维护含 Misha Khalman 等）
- **链接：** https://github.com/anthropics/skills/tree/main/skills/claude-api
- **许可：** 见仓库 `LICENSE.txt`（Complete terms）
- **入库日期：** 2026-09-30
- **一句话说明：** Anthropic 官方 Claude API / SDK 参考技能：模型 ID、定价、流式、工具/MCP、缓存、迁移与 **build-eval / hillclimb / cost-optimize / prompt-audit** 等子命令，把平台最佳实践编译进 `SKILL.md` 与 `shared/` 指南。
- **为什么值得保留：** 与本站 coding agent 评测（[RLE-Bench](../../wiki/entities/rle-bench.md)）、真机 autoresearch（[ENPIRE](../../wiki/methods/enpire.md)）及 [Agent Skills 生态](../../wiki/entities/agent-skills-addyosmani.md) 直接相关；是 **API 漂移**（adaptive thinking、web_search 工具版本等）的权威对照。
- **沉淀到 wiki：** 是 → [`wiki/entities/anthropic-claude-api-skill.md`](../../wiki/entities/anthropic-claude-api-skill.md)
- **代码：** **已开源**（GitHub 公开仓库；无独立「项目页」，以 `SKILL.md` + `shared/*.md` 为可运行规约）

---

## 触发与边界（SKILL.md 摘要）

- **何时加载：** 任务涉及 Claude/Anthropic 模型、SDK、agent/MCP、工具定义、RAG、LLM-as-judge、流式/缓存/定价等；**禁止**在未 grep 确认前对 OpenAI/Gemini 等项目误用本技能。
- **输出约束：** 默认 **官方 Anthropic SDK**（非 OpenAI 兼容 shim）；模型默认 **`claude-opus-5-5`** + **`thinking: {type: "adaptive"}`** + 长任务默认 streaming。
- **API 漂移表（2025–2026）：** `budget_tokens` 在 Opus 5.5 / Fable 5 等会 **400**；web 工具类型升级；Files/Skills API 出 beta；PHP 参数 camelCase 等——以技能内 `{lang}/` 文件为准。

## 子命令（Slash / 裸子命令字符串）

| 子命令 | 作用 |
|--------|------|
| `migrate` | 按 `shared/model-migration.md` 迁移到更新 Claude 模型（含 prompt 审计） |
| `prompt-audit` | 扫描 CLAUDE.md、skills、工具描述中的 **陈旧 ritual / 矛盾规则** |
| `upgrade` | Python SDK 大版本升级（如 `anthropic` 0.x→1.x） |
| `cost-optimize` | 用量画像 + 缓存/批处理/effort/模型档位；可选 eval 测质量 tradeoff |
| `build-eval` | 访谈式构建 eval：采样输入、选 grader、可运行 runner + 成本预估 |
| `hillclimb` | 对已有 eval 迭代改 prompt/skills/harness；train/test 防过拟合 |
| `preserved-thinking-migration` |  preserved thinking 前缀一致性迁移 |

## 仓库结构（技能包）

```
skills/claude-api/
├── SKILL.md              # 入口、子命令表、语言检测、Surface 选型
├── shared/               # evals、cost、migration、prompt-audit 等长指南
├── python/ typescript/ java/ go/ ruby/ csharp/ php/ curl/
└── LICENSE.txt
```

## 关联官方博文（已单独归档 sources/blogs）

- [Demystifying evals for AI agents](../blogs/anthropic_demystifying_evals_ai_agents_2026-01-09.md)
- [Reducing cost and performance with Claude Platform](../blogs/anthropic_claude_platform_cost_performance_2026-09-08.md)
- [Getting the most out of Opus 5.5](../blogs/claude_dev_opus_5_5_guide_2026-09-22.md)
- [Automating eval design and hillclimbing](../blogs/claude_dev_automating_eval_hillclimbing_2026-09-28.md)
