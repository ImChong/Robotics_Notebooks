---
type: entity
project_id: jeremykamber-strata-tiered-memory
tags: [llm-agents, agent-memory, sqlite, python, memory-lifecycle]
status: complete
updated: 2026-11-20
code: https://github.com/jeremykamber/strata-memory
original:
  - ../../sources/repos/jeremykamber-strata-memory.md
  - ../../sources/blogs/jeremykamber-strata-tiered-memory-2026-05-05.md
related:
  - ./kytheros-strata-mcp-memory.md
  - ./paper-strata-experience-graphs.md
  - ../concepts/agent-memory-and-dreaming.md
sources:
  - ../../sources/repos/jeremykamber-strata-memory.md
  - ../../sources/blogs/jeremykamber-strata-tiered-memory-2026-05-05.md
summary: "Jeremy Kamber 的 Strata Memory 以 Markdown 活跃层和降温层、JSON 冷归档与 SQLite 影子索引实现长期记忆生命周期，使用规则 Janitor 迁移并可选 LLM 蒸馏。"
---

# Strata Memory：活跃、降温、归档三层记忆

作者：[Jeremy Kamber](https://jeremykamber.com/blog/strata-a-tiered-memory-system-for-effective-ai-agents)；官方实现：[jeremykamber/strata-memory](https://github.com/jeremykamber/strata-memory)。

## 一句话定义

通过文件年龄、访问频率和轻量索引，把 Agent 的即时记忆、近期知识和长期归档分配到不同存储层，而不是永久塞进同一个检索库。
## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| LLM | Large Language Model | 可选蒸馏器使用的语言模型 |
| FTS5 | Full-Text Search 5 | 冷归档影子索引的关键词检索 |
| LRU | Least Recently Used | 根据最近访问情况决定降级/回收 |
| JSON | JavaScript Object Notation | 冷归档序列化存储 |

## 核心机制

| 层 | 当前代码行为 |
|---|---|
| Active | `active/` 下直接可读写的 Markdown 与 `index.md` |
| Cooled | `cooled/` 下只读检索的 Markdown，跟踪访问次数 |
| Archive | `archive/` JSON 冷存储 + SQLite FTS5 Shadow Index；命中可再水合 |
| Janitor | 按时间和访问计数移动、驱逐、回升文件，不需要模型参与迁移判断 |
| Distiller | **可选**的后台 LLM 会话事实提取；并非每次查询执行 |

README 示例默认项目 14 天、实体 60 天、任务 7 天后可能降温；访问可触发回升。阈值可配置，不应当作普适最优值。

## 流程总览

```mermaid
flowchart LR
  Input["Agent 记忆写入"] --> Active["active/ Markdown"]
  Active -->|达到时效阈值| Janitor["Janitor 规则迁移"]
  Janitor --> Cooled["cooled/ Markdown"]
  Cooled -->|长期未访问| Archive["archive/ JSON"]
  Archive --> Index[("SQLite FTS5 Shadow Index")]
  Query["Agent 查询"] --> Active
  Query --> Cooled
  Query --> Index
  Index -->|命中恢复| Active
  Cooled -->|访问频繁回升| Active
  Transcript["会话转写"] --> Optional["可选 Distiller"]
  Optional --> Active
```

## 源码运行时序图

已核对 `strata/_cli_main.py` 注册表和 `strata/janitor.py`：

```mermaid
sequenceDiagram
  autonumber
  actor User as 用户 / Agent
  participant CLI as strata/_cli_main.py
  participant Jan as strata/janitor.py Janitor
  participant S1 as Stratum1Storage
  participant S2 as Stratum2Storage
  participant S3 as Stratum3Storage
  User->>CLI: strata maintenance
  CLI->>Jan: 执行生命周期处理
  Jan->>S1: scan_stale_files()
  S1-->>Jan: 过期候选
  Jan->>S2: 复制 Markdown 并建立访问记录
  Jan->>S1: 删除已迁移文件 / 重建索引
  Jan->>S2: 读取 LRU 驱逐候选
  Jan->>S3: archive_file(JSON + shadow index)
  Jan->>S2: 删除已归档 Markdown / 访问记录
  Jan-->>User: 返回迁移、驱逐结果
```

可用 `pip install git+https://github.com/jeremykamber/strata-memory.git` → `strata init` → `strata add/search/maintenance` 试用。上述运行序列按已读源码归纳，并未本次实跑。

## 早期博客与当前代码区别

作者 2026-05-05 博客描述**计划中的中层**为 PostgreSQL/pgvector `agent-db`，以及迁移时 LLM 压缩；当前开源 README 和 Janitor 代码则使用 `cooled/` Markdown + 规则搬运，LLM Distiller 是可选、独立的会话提取组件。**不能把博客设计直接当成当下实际代码。**

## 局限与机器人启发

冷热规则简单且可审计，但固定时效不等于知识真伪、任务价值或危险程度；关键控制参数、安全规则不应仅因少访问而隐藏。此项目不是机器人记忆性能测评。

## 关联页面

- [Kytheros Strata MCP](kytheros-strata-mcp-memory.md) — 侧重质量门、跨 CLI 检索。
- [STRATA 经验图](paper-strata-experience-graphs.md) — 侧重结构归纳与负迁移防护。
- [Agent 记忆与 dreaming](../concepts/agent-memory-and-dreaming.md) — 以重写/纠错对比分层迁移。

## 参考来源

- [官方 README / Janitor 源码归档](../../sources/repos/jeremykamber-strata-memory.md)
- [作者博客早期架构归档](../../sources/blogs/jeremykamber-strata-tiered-memory-2026-05-05.md)

## 推荐继续阅读

- [当前开源实现](https://github.com/jeremykamber/strata-memory) · [作者 2026-05-05 原文](https://jeremykamber.com/blog/strata-a-tiered-memory-system-for-effective-ai-agents)。
