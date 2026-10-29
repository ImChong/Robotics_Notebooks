# The new rules of context engineering for Claude 5 generation models

> 来源归档

- **类型：** blog（claude.dev / Anthropic）
- **作者：** Thariq Shihipar
- **链接：** https://claude.dev/blog/the-new-rules-of-context-engineering-for-claude-5-generation-models/
- **发布日期：** 2026-07-24
- **入库日期：** 2026-10-28
- **一句话说明：** Claude Code 团队分享如何随模型能力变化调整系统提示、工具描述、记忆、技能和参考材料。
- **沉淀到 wiki：** 补充 → [Agentic Coding 时代的软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)

## 核心做法与边界

- Anthropic 报告针对 Opus 5 和 Fable 5 删除 Claude Code 系统提示的 80% 以上，编码评测未出现可测量的损失。这是特定系统、模型和评测的结果，不等于所有上下文或用户规则都应删去 80%。
- 作者建议减少互相冲突、重复或过度约束的规则；用清晰的工具接口与描述、按需加载的技能、记忆及丰富参考代替一味堆叠指令。
- 仓库说明保留特有陷阱和关键约束，长技能按需拆分；重要领域仍可能需要严格规则。上下文改动应使用实际任务和评测确认效果。

文章为内部工程经验，未提供与该次精简直接配套的公开实验仓库。
