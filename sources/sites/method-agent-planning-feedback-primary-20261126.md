# Anthropic：Building effective agents

- **URL：** <https://www.anthropic.com/research/building-effective-agents>
- **类型：** 官方文档或作者工程文章
- **核验日期：** 2026-11-26
- **通读：** 已完成
- **核对范围：** What are agents?、When (and when not) to use agents、Prompt chaining、Orchestrator-workers、Evaluator-optimizer、Agents 的正文及流程图。

## 来源摘要

文章区分预定义路径的 workflow 与动态选择步骤的 agent；任务执行应从工具或环境获得客观反馈，可在检查点由人介入，并设置最大迭代数。评价与优化循环在有明确评价条件、反馈可带来可测改善时更合适。文章提供设计经验，不是各架构通用胜率基准。

关联：[方法页](../../wiki/methods/agent-planning-feedback.md)。
