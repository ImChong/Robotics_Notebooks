# github/spec-kit

- **类型：** repo
- **标题：** GitHub Spec Kit
- **链接：** https://github.com/github/spec-kit
- **官方文档：** https://github.github.io/spec-kit/
- **核查日期：** 2026-10-13
- **开源状态：** 已开源；公开仓库提供 Specify CLI、模板、agent 集成与文档。
- **许可证：** 以仓库当前 LICENSE 文件为准；第三方 agent / 模型服务有各自条款。
- **一句话说明：** GitHub 维护的开源工具包，将面向编码代理的 SDD 具体化为 constitution、specify、clarify、plan、tasks、analyze、implement、converge 等阶段。

## 核心入口

- **典型路径：** 项目原则 / constitution → feature spec → 澄清需求 → 技术 plan → task 列表 → 跨工件一致性检查 → 实现 → convergence。
- **重要机制：** analyze 用于检查 spec、plan、tasks 之间的冲突、遗漏和任务追溯关系；失败时回到负责该问题的阶段修正规格或计划。
- **规格演化：** 官方文档提醒，在功能理解变化后要把仍有效的决策回写到正确工件；规格不是一次写完后冻结的瀑布式签字文档。
- **边界：** Spec Kit 是一种 SDD 工具实现，不等于 SDD 唯一标准；它不能替团队定义业务正确性，也不能自动证明代码满足所有非功能属性。

## 对 wiki 的映射

- [规格驱动开发（SDD）](../../wiki/concepts/specification-driven-development.md)
- [SDD / TDD / BDD / ATDD 对照](../../wiki/comparisons/sdd-tdd-bdd-development-methods.md)

## 参考来源

- [GitHub Spec Kit README](https://github.com/github/spec-kit)
- [Spec Kit 官方 SDD 概念说明](https://github.github.io/spec-kit/concepts/sdd.html)
- [Agentic SDD 参考流程](https://github.github.io/spec-kit/reference/agentic-sdd.html)
