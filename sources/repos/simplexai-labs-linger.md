# simplexai-labs/linger：本地个人 AI Agent

- **类型：** repo
- **原始链接：** https://github.com/simplexai-labs/linger
- **入库日期：** 2026-11-20
- **核对范围：** README 全文、docs/architecture.md 与 docs/wiki-and-memory.md 的核心架构/知识索引章节及根目录运行入口
- **通读：** 已完成

## 原始资料要点（归纳，非全文转载）

- 个人 Agent：Room/Chat 触发 Computer Daemon 与本地 Runtime；模型经 LLM Gateway，电脑动作通过 Guardrail 后交给 Cua、Playwright 或 Shell。
- 为用户维护跨 Agent 可搜索的 Personal Wiki（Client/Contact/Project/Topic/Session 实体页）及每 Agent 自己的会话记忆；标注事实来源与变更日志。
- 模型推理可对接本地 PolyStrata/Strata 等后端；Linger 属于 agent 应用层，不能当作推理引擎或单一记忆库。
- README 给出桌面安装或 Docker Compose 部署；连接邮箱等个人数据有隐私、授权与权限隔离风险。

## 对 wiki 的映射

- [simplexai-labs/linger：本地个人 AI Agent](../../wiki/entities/linger-personal-agent.md)
