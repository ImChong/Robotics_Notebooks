# Getting the most out of Opus 5.5 in Claude and Claude Code

> 来源归档

- **标题：** Getting the most out of Opus 5.5 in Claude and Claude Code
- **类型：** blog（claude.dev）
- **作者：** Addy Osmani
- **链接：** https://claude.dev/blog/getting-the-most-out-of-opus-5-5/
- **发布日期：** 2026-09-22
- **入库日期：** 2026-09-30
- **一句话说明：** Opus 5.5 使用指南：一次消息定义「完成标准」并长跑、删除「think hard」类指令、CLAUDE.md 规定何时停/何时继续、子 agent 并行审计、TASKS.md 抗 context 压缩；视觉/长文档/表格产出改进；Fable 级 bio/cyber  safeguard 触发的模型切换说明。
- **沉淀到 wiki：** 交叉 → [`wiki/entities/anthropic-claude-api-skill.md`](../../wiki/entities/anthropic-claude-api-skill.md)、[`wiki/entities/agent-skills-addyosmani.md`](../../wiki/entities/agent-skills-addyosmani.md)

---

## Claude Code 要点

- **整任务 + 完成定义：** 多步仓库级迁移/测试通过；可 mid-run 追加约束（Enter 插队）。
- **去掉 think carefully / step by step：** 模型默认 adaptive thinking；简单问句可写「Answer directly」或调 **effort**。
- **CLAUDE.md：** 非阻塞步骤「状态与下一动作同条消息」；仅不可解释失败或破坏性操作前询问；destructive 权限仍开。
- **子 agent：** 大 audit/migration 按服务 fan-out + 验证据 + 汇总表。
- **TASKS.md：** 长 run 任务清单文件化，避免仅靠 scrollback。
- **收尾：** 先读「Blocked on me / needs from you」；PR review prompt 强调 file:line 与复现。
- **设计：** 列 **不要** 的 UI 模式（cream 背景、01/02/03 标签等）比泛泛「不要 generic」有效。
- **/fast：** 交互式 research preview，更快 token 到达，额外用量与单价。

## Claude 应用

- 图表/截图 **附件** 而非手打数字；长文档/ deck 自相矛盾检查；直接要可分享 xlsx/docx。
- Project 指令：已答问题视为 settled（长分析项目可例外）。

## Safeguard 切换

- Apps：flag 后切旧模型；新 chat 或关「Switch models when flagged」。
- Code：`/model` 切回、Esc×2 改消息、`/feedback`。
- 勿要求模型在回复中 **复现内部 reasoning**（易触发 flag 类）。
