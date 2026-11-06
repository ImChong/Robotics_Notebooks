# The new rules of context engineering for Claude 5 generation models（通读）

> 来源归档（原文通读 · 支撑 `wiki/concepts/agentic-coding-software-fundamentals.md`）

- **标题：** The new rules of context engineering for Claude 5 generation models
- **作者：** Thariq Shihipar（Anthropic）
- **类型：** blog / coding-agents / context-engineering
- **原文：** <https://claude.dev/blog/the-new-rules-of-context-engineering-for-claude-5-generation-models/>（2026-11 抓取页面全文，发布日期 2026-07-24，约 7 分钟阅读）
- **一句话说明：** Claude Code 团队总结：随模型能力提升，系统提示、CLAUDE.md、技能中的大量硬规则与示例已成负担，应转向判断力、接口设计、渐进披露与丰富参考。
- **核对范围：** 通读全文：Unhobbling Claude、Then and now 六组对照、Applying this to your context（System prompt / CLAUDE.md / Skills / References）、Try simplifying 全部章节；文末 Related posts 仅为链接列表，未阅读所列文章。文中提到的 Fable field guide 与 claude doctor 实现细节未另行核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **结果：** 针对 Claude Opus 5 与 Claude Fable 5，删除了 Claude Code 系统提示的 80% 以上，编码评测无可测量损失（特定系统、模型与评测，非通用结论）。
- **过度约束：** 系统提示、CLAUDE.md、技能与用户请求中常出现相互冲突的指令（如"按需保留文档"与"不要加注释"），模型须额外权衡。
- **六组对照：** 给规则 → 让模型用判断（例：改为"写得像周围代码，匹配其注释密度、命名与惯用法"）；给示例 → 设计接口（示例会把探索空间框死；如 Todo 工具仅靠 status 枚举 pending/in_progress/completed 与"只保持一个 in_progress"提示用法）；全部前置 → 渐进披露（验证与代码评审拆为技能；部分工具延迟加载，经 ToolSearch 取完整定义）；重复强调 → 简洁工具描述（用法写在工具描述而非系统提示）；CLAUDE.md 记忆 → 自动记忆；简单规格 → 丰富参考（HTML artifact、测试套件、待移植函数、评审 rubric 与验证 agent）。
- **上下文装配建议：** 系统提示与产品语境绑定，自建 harness 才需重点投入；CLAUDE.md 保持精简，篇幅花在代码库特有的陷阱上，不写看文件系统即知的显然内容；技能是轻量指引，仅在重要领域才强约束，长技能拆成文件树；参考优先用代码形态（HTML 原型优于文字描述或截图）。
- **工具：** `claude doctor`（`/doctor`）用于自动精简技能与 CLAUDE.md。

## 对 wiki 的映射

- [`wiki/concepts/agentic-coding-software-fundamentals.md`](../../wiki/concepts/agentic-coding-software-fundamentals.md)
