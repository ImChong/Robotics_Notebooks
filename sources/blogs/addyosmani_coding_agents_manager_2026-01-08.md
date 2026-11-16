# Your AI coding agents need a manager / 你的编码代理需要经理

> Addy Osmani 原创工程实践博文归档（摘要与结构化摘录，非原文转载）

- **类型：** blog
- **作者：** Addy Osmani
- **标题：** Your AI coding agents need a manager
- **链接：** https://addyosmani.com/blog/coding-agents-manager/
- **发布日期：** 2026-01-08
- **入库日期：** 2026-11-16
- **核对范围：** 通读文章的双模式工作流、任务简报、授权边界、验证闭环、异步检查点、worktree 隔离、WIP 限额、合并次序与复盘建议。
- **通读：** 已完成
- **沉淀到 wiki：** [多 Agent 编码管理与认知把关](../../wiki/concepts/coding-agent-orchestration-and-cognitive-oversight.md)
- **关联概念：** [Agentic Coding 时代的软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)

## 主要观点（作者的工程判断）

1. **把 coding agents 当异步团队管理。** 人从逐条提示转向确定「该不该做、谁负责、怎么证明完成」；并行产出很快，人类审查注意力反而成为瓶颈。
2. **双模式：** 架构、模糊产品决策和高风险修改由人紧密参与；边界清晰的测试、文档、迁移、局部修复可以异步交付。
3. **任务简报：** 目标、仓库上下文、约束、非目标、验收标准、文件所有权与验证计划应前置；用既有代码和仓库规则作参考。
4. **验证：** 执行测试、lint、类型检查，附 PR 证据包；必要时实现 agent 与独立评审 agent 分工，复查后再集成。
5. **协调：** worktree/branch 隔离；一个 agent 对应一个 PR；尽量避免多人修改同一文件；维护 WIP 限额和放弃标准；异步状态报告包含进展、下一步、风险、阻塞。
6. **迭代：** 将失败模式写回 \`AGENTS.md\` 与检查清单，但无需将所有任务都启动成十几个并行 agent。

## 事实边界

文中的「四至五个后台 agent」等数字是作者个人工作流描述，不是经对照实验确定的最优并行度；本文没有提供可复现代码或性能基准。组织层面的建议需结合团队 review 容量与安全约束验证。
