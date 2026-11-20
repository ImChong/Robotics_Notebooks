# jeremykamber/strata-memory：分层文件记忆实现

- **类型：** repo
- **原始链接：** https://github.com/jeremykamber/strata-memory
- **入库日期：** 2026-11-20
- **核对范围：** README 全文、strata/_cli_main.py 命令注册与分发、strata/janitor.py 迁移/回收源码
- **通读：** 已完成

## 原始资料要点（归纳，非全文转载）

- 当前仓库实现将记忆分成 active/ Markdown、cooled/ Markdown、archive/ JSON + SQLite FTS5 shadow index。
- Janitor 按文件修改时间、LRU 和访问次数执行降温、归档、回升；源码调用 Stratum1Storage/2Storage/3Storage。
- 官方安装为 pip install git+https://github.com/jeremykamber/strata-memory.git；strata init、add、search、maintenance 可在 CLI 使用。
- 可选 Distiller 使用 LLM 从会话转写提取事实，和无需 LLM 的 Janitor 生命周期管理分开。
- 早期作者博客写到 PostgreSQL/pgvector 与迁移时 LLM 压缩；**当前 README/Janitor 源码**的基本层级是文件/SQLite，不应把博客规划当作当前已实现能力。

## 对 wiki 的映射

- [jeremykamber/strata-memory：分层文件记忆实现](../../wiki/entities/jeremykamber-strata-tiered-memory.md)
