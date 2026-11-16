# Best practices for Claude Code / Claude Code 最佳实践

> Anthropic 官方产品文档的读取索引与工程要点，不是源码仓库或独立实验报告

- **类型：** site（官方操作文档）
- **机构：** Anthropic
- **标题：** Best practices for Claude Code
- **链接：** https://code.claude.com/docs/en/best-practices
- **版本/访问日期：** 2026-11-16 在线文档当前可见版（无独立静态版本号；可能持续更新）
- **入库日期：** 2026-11-16
- **核对范围：** 通读文档从「Deterministic and non-deterministic instructions」到「What you can stop doing」的全部主体章节：特定 prompt、CLAUDE.md、路径规则与 skills、验证/Stop hook、权限与 sandbox、subagent、成本/上下文、无值守运行、并行 worktree 与过时建议。
- **通读：** 已完成
- **沉淀到 wiki：** [多 Agent 编码管理与认知把关](../../wiki/concepts/coding-agent-orchestration-and-cognitive-oversight.md)
- **代码状态：** 文档不是独立源码项目，无可运行仓库入口；示例命令与配置依赖 Claude Code 产品环境。

## 文档原文的核心边界

| 层级 | 适用手段 | 保障的性质 |
| --- | --- | --- |
| 意图和偏好 | 具体 prompt、精简的 `CLAUDE.md`、按路径分配的 rules、按需调用的 skills | 模型参考这些信息，但它们**不是必执行保证** |
| 硬性规则 | command/http hooks、权限 deny、sandbox、管理设置、CI、人工审批 | harness 或外围系统执行；必要时阻止操作或合并 |
| 验收与独立复核 | 测试/类型检查/lint、截图、`/verify`、独立 code review | 从「看起来完成」变成可核验的结果 |
| 异步/并行 | `--max-turns` / `--max-budget-usd`、分支/worktree、隔离权限、PR 审批 | 管理资源、工作区冲突与人的最终签署 |

## 原文强调的取舍

- `CLAUDE.md` 不必复述 agent 能自行在仓库发现的信息；团队独有流程宜存短规则或按需 skills。
- 若某条禁令必须无例外执行，不应仅靠增强提示词；使用配置、权限及确定性 hook。文档区分 command/http hook 与借助模型判断的 prompt/agent hook。
- 按规格审查需求是否落实，而不是要求第二个 agent 不加区分地挑风格问题。
- 无值守前回答五个问题：完成条件、验证方式、可访问范围、运行上限、交付位置及谁审批。
- 并行任务最好使用独立 worktree；subagent 不是天然更聪明的模型，主要价值在隔离上下文、并行与独立审阅。
- 文档介绍的具体默认模型、权限模式与新命令是**访问时刻的产品状态**，以后应回看官方原文，不宜视为永久不变的最佳实践。
