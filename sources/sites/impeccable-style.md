# Impeccable（impeccable.style）

- **标题：** Impeccable — The missing design vocabulary for agents
- **类型：** site / project-page
- **URL：** <https://impeccable.style>
- **入库日期：** 2026-09-30
- **代码：** <https://github.com/pbakaus/impeccable>（归档 [`sources/repos/pbakaus-impeccable.md`](../repos/pbakaus-impeccable.md)）
- **开源状态：** **已开源**（Apache-2.0；skill + CLI 引擎；detector 为确定性规则）

## 一句话摘要

开源 **设计语言 + 命令面 + 检测器**： **`/impeccable <command>`** 共 **24 条**共享命令（`polish`、`typeset`、`distill`、`audit` 等），配合 **`PRODUCT.md` / `DESIGN.md`** 持久化产品与视觉系统，并用 **61 条确定性 detector 规则**（无 LLM、无 API key）在 hook / PR / Chrome DevTools 中抓 AI slop。

## 公开信息要点（截至入库日 2026-09-30）

- **安装：** `npx impeccable install`（项目根）；Claude `/plugin marketplace add pbakaus/impeccable`；或 `npx skills add pbakaus/impeccable`。
- **初始化：** `/impeccable init` 写 **PRODUCT.md**（用户、目的、a11y 等 durable truth）；`/impeccable document` 从代码抽 **DESIGN.md**。
- **自动化：** `/impeccable hooks on` — PostToolUse / Stop hook 把 detector 发现反馈给 agent；GitHub Copilot 实验内置。
- **Live 模式：** `/impeccable live` — 浏览器内点选元素迭代；`/impeccable generate` 变体生成。
- **设计方向库：** 站点提供 human-reviewed **design directions**（如 Neo Mirai case study）。
- **血缘：** README 明示自 Anthropic [frontend-design](https://github.com/anthropics/skills/tree/main/skills/frontend-design) 演进而来。

## 为何值得保留

- 把「审美」拆成 **可重复命令 + 可执行检测**，适合与本站 `docs/` 静态站迭代、`make ci-preflight` 式 **质量栏写进文件** 的文化对齐。
- 与 Taste Skill 的 **纯 SKILL 约束**、Skillry 的 **成品市场** 构成三角选型（见 comparison 页）。

## 关联资料

- 代码归档：[`sources/repos/pbakaus-impeccable.md`](../repos/pbakaus-impeccable.md)
- wiki：[`wiki/entities/impeccable.md`](../../wiki/entities/impeccable.md)
