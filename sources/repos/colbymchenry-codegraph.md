# CodeGraph（colbymchenry/codegraph）

> 来源归档

- **标题：** CodeGraph
- **类型：** repo
- **作者：** Colby McHenry
- **代码：** <https://github.com/colbymchenry/codegraph>
- **项目页：** <https://colbymchenry.github.io/codegraph/>
- **包：** npm `@colbymchenry/codegraph`
- **许可：** MIT
- **入库日期：** 2026-10-14
- **一句话说明：** 本地优先的代码智能工具：用 Tree-sitter 和静态解析构建代码符号、调用与依赖图，存入本地 SQLite，并通过 MCP / CLI 向多个 coding agent 提供代码上下文、调用路径与变更影响分析；文件变化后自动增量同步。
- **为什么值得保留：** CodeGraph 代表 coding agent 的一种代码库理解基础设施：先把仓库结构预索引为显式关系图，再让 agent 按问题检索代码和路径。它与本库已有的 [Graphify](../../wiki/entities/graphify.md) 和 [Understand Anything](../../wiki/entities/understand-anything.md) 同属代码知识图谱工具，但主路径更集中于 AST/静态关系、MCP 查询和自动同步。
- **开源状态：** 已开源，仓库以 MIT 许可发布；项目页链接至官方仓库。
- **沉淀到 wiki：** 是 → [`wiki/entities/codegraph.md`](../wiki/entities/codegraph.md)

## README 要点（归纳）

- **安装与初始化：** `codegraph install` 为支持的 coding agent 配置 MCP，在项目目录运行 `codegraph init` 建图；也可通过 npm 包安装或直接调用 CLI。
- **建图与存储：** 官方说明称 Rust 内核使用 Tree-sitter 等解析路径抽取函数、类、方法及调用、导入、继承、实现关系，再解析跨文件引用。索引存于 `.codegraph/codegraph.db` 本地 SQLite，并使用 FTS5 全文索引。
- **查询：** MCP 默认提供 `codegraph_explore`，返回匹配符号的源码、调用路径和影响范围；CLI 另有 `query`、`callers`、`callees`、`impact`、`affected` 等命令。
- **同步：** 本地文件事件触发 watcher，经防抖后增量更新；MCP 重新连接时会对工作树执行同步核对。沙箱或关闭 daemon 时可手动同步。
- **覆盖：** 项目页列出 Python、C/C++、CUDA、Rust 等语言以及框架路由、动态分派和 iOS / React Native 跨语言桥接的专门规则。语言状态和边界以官方支持列表为准。
- **平台与 Agent：** 官方 README 列出 Windows、macOS、Linux 的 x64 / arm64 构建，并提供 Claude Code、Cursor、Codex CLI、OpenCode、Hermes、Gemini CLI、Antigravity、Kiro 和 GitHub Copilot 的安装配置。
- **隐私与遥测：** 代码图存放在本机。官方声明会发送汇总的匿名使用统计（如命令与语言计数），不包含代码、文件路径、符号名或查询；可通过 CLI、`CODEGRAPH_TELEMETRY=0` 或 `DO_NOT_TRACK=1` 关闭。
- **性能证据：** README 报告其对 7 个开源代码库的 Claude Opus 4.8 测试中，平均减少 88% 工具调用、62% token；这是项目方基准，依赖其问题集、模型与测试流程，不能直接视为所有仓库和任务的普遍结果。

## 对 wiki 的映射

- 沉淀为 [`wiki/entities/codegraph.md`](../wiki/entities/codegraph.md)，并与 [Graphify](../../wiki/entities/graphify.md)、[Understand Anything](../../wiki/entities/understand-anything.md) 对照。
