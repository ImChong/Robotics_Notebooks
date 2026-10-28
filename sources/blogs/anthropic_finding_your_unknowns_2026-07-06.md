# A field guide to Claude Fable 5: Finding your unknowns

> 来源归档

- **类型：** blog（Claude / Anthropic）
- **作者：** Thariq Shihipar
- **链接：** https://claude.com/resources/articles/a-field-guide-to-claude-fable-finding-your-unknowns
- **发布日期：** 2026-07-06
- **入库日期：** 2026-10-28
- **一句话说明：** 通过实施前、中、后的迭代，发现人和 coding agent 对目标、代码及约束的未知。
- **沉淀到 wiki：** 补充 → [Agentic Coding 时代的软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)

## 核心做法

- 区分已知的已知、已知的未知、未说出的已知、未知的未知；计划不能消除实施过程中出现的新未知。
- 实施前让 agent 查盲点、头脑风暴或做廉价原型；对改变架构的歧义逐一访谈，并提供代码等具体参考，审阅实施计划。
- 实施中记录边界情况和偏离计划的决定；完成后用解释、演示和提问帮助人理解变更。
- 过度具体可能阻碍转向，过于模糊则可能让模型自行选择不适用的默认方案。模型提出的盲点仍需对照代码和实际约束核验。

文章是作者使用 Claude Code 的经验，不是关于所有模型的受控实验；未提供配套项目仓库。
