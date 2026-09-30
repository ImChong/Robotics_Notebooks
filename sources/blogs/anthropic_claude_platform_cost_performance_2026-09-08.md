# Reducing cost and improving performance with Claude Platform

> 来源归档

- **标题：** Reducing cost and improving performance with Claude Platform
- **类型：** blog（Claude / Anthropic）
- **作者：** Lance Martin
- **链接：** https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform
- **发布日期：** 2026-09-08（文内后续补测 Opus 5.5，2026-09-22 发布）
- **入库日期：** 2026-09-30
- **一句话说明：** 三杠杆降本不降质：**prompt cache**（前缀字节一致、defer_loading 工具、系统更新用 message、预 warm）、**prompt-audit 去反模式**（verify twice、scratchpad ritual、矛盾规则、过时 thinking 配置）、**effort 校准**（强模型低 effort 可能更便宜）；`/claude-api cost-optimize` 与 `hillclimb` 自动化搜索。
- **沉淀到 wiki：** 交叉 → [`wiki/entities/anthropic-claude-api-skill.md`](../../wiki/entities/anthropic-claude-api-skill.md)

---

## Prompt cache 要点

- 缓存绑定 **模型**；前缀须 **字节级一致**；TTL 默认 5 分钟（长工具/子 agent 阻塞可改 1h）。
- 避免：中途改 effort/thinking（Opus 5 / Fable 5.1 部分可 mid-conversation 改 effort）、前缀放动态 timestamp、工具定义重排、fork 前缀不一致、同步调用超过 TTL。
- 修复：Console/API **cache miss 诊断**、`defer_loading` 稀有工具、系统变更用 **message**、静态前缀在前动态在后、compaction 时顺带换模型/effort、`max_tokens:0` 预 warm。

## Instructions 反模式（prompt-audit）

- 验证仪式、强调 booster、强制 scratchpad、陈旧 few-shot、矛盾退款规则、固定 budget thinking（Opus 5.5 会拒或行为异常）。
- 案例：Opus 4.8→5.5 仅换模型 ID 约 **-18% 成本**；清反模式再 **-9%** 且准确率 **+2pt**。

## Effort

- 非「越高越好」：HLE 上 max effort 增益可能落在噪声内。
- **强模型 + low effort** 可击败弱模型 + high effort（CursorBench 3.2：Fable 5.1 low ≈ Fable 5 high，约 1/3 成本）。
- `hillclimb`：客服 benchmark 从 Opus 4.8 high → Opus 5 low + audit → Sonnet 5 low + 路由规则；hold-out **90.5% vs 78.6%**，成本约 **1/5**。

## cost-optimize 四基准（Opus 5.5 基线）

LegalBench ~**-67%**；tau2 retail ~**-73%**；OfficeQA Pro ~**-72%**；SWE-bench Verified ~**-24%**（多为 effort medium + 输出约束）。

## 子命令入口

- `/claude-api prompt-audit` — 迁移后扫 prompt/skills/工具描述  
- `/claude-api cost-optimize` — 组织用量或 `usage` 日志 + 可选 eval  
- `/claude-api hillclimb` — train/test 迭代
