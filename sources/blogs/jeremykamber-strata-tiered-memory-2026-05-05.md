# Jeremy Kamber：Strata: A Tiered Memory System for Effective AI Agents

- **类型：** blog
- **原始链接：** https://jeremykamber.com/blog/strata-a-tiered-memory-system-for-effective-ai-agents
- **入库日期：** 2026-11-20
- **核对范围：** 作者 2026-05-05 博客正文（Active、Transition、Medium-term、Archive、System Integration、Synthesis 全节）
- **通读：** 已完成

## 原始资料要点（归纳，非全文转载）

- 博客提出三层记忆：当前 Markdown 活跃层、Postgres/pgvector agent-db 中期层、JSON 冷归档及 SQLite Shadow Index。
- 旧文件先按时间触发，再由 LLM 压缩成 Memory Block；中期记忆按 LRU 回收；冷层通过关键词影子索引恢复。
- 与目前 GitHub 实现存在方案演进差异：当前项目的 cooled/ 仍为 Markdown，Janitor 无需 LLM；不能按该博客推断最新版必需 Postgres。

## 对 wiki 的映射

- [Jeremy Kamber：Strata: A Tiered Memory System for Effective AI Agents](../../wiki/entities/jeremykamber-strata-tiered-memory.md)
