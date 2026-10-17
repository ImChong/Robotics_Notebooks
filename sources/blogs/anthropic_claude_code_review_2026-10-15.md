# How the Claude Code team does code review

> 来源归档

- **标题：** How the Claude Code team does code review
- **类型：** blog（claude.dev / Anthropic）
- **作者：** Addy Osmani、Morgan Westlee Lunt、Qing Wang
- **链接：** https://claude.dev/blog/solving-code-review-at-anthropic/
- **发布日期：** 2026-10-15
- **入库日期：** 2026-10-17
- **一句话说明：** Claude Code 团队介绍如何把代码审查改造成持续、多层的风险控制流程：让 Claude 和自动化检查处理常规问题，把人工注意力留给高风险变更、方向与责任判断。
- **沉淀到 wiki：** 补充 → [Agentic Coding 时代的软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)

## 核心做法

- **把验证前移到 PR 之前。** 先让 Claude 运行测试、构建或界面核验，并展示结果；再由拥有新上下文的子 agent 检查 diff。重复出现的审查问题沉淀到 CLAUDE.md、REVIEW.md 或 lint 规则。
- **在每个 PR 上做 AI 评审和安全检查。** 多个 agent 并行找问题，另一个 agent 组核实发现；团队可按 Quick、Standard、Deep 调节评审强度。安全与缺陷评审中的问题需要在合并前处理或说明。
- **让人工审批随变更风险变化。** 文档等低风险改动在自动检查通过后可不要求人工审批；普通改动需要一人；高风险改动通常需要两人。作者不能决定自己的风险等级，CI 等敏感路径始终需要人工把关，agent 的批准不代替人的批准。
- **用队列与渐进发布降低组合风险。** 合并队列对合入 main 后的组合状态重跑检查；功能开关、内部先用和分阶段发布让问题更容易发现与回退。
- **限制审查循环并持续校准。** 把事故和回滚案例加入评测集，跟踪误报、人工审查时间与多轮返工；限制重复轮次，对后续变更优先复查新增 diff。

## 报告结果与边界

Anthropic 报告，2026 年 6 月至 9 月，Claude Code 仓库合并 PR 数量超过翻倍，而需要回滚的 PR 占比下降超过 40%。这是团队对自身流程的报告；文章目标是减少每次变更消耗的人工审查时间，同时将回滚和事故率作为质量护栏。低风险自动放行属于 Anthropic 内部仓库策略，不能直接套用到其他团队，尤其是会影响真机安全、权限、支付或发布链路的变更。

文章中的 /code-review 可在本地运行较轻量的审查；Anthropic 还介绍了 Claude Code Team 和 Enterprise 上的 Code Review 产品能力。本文是工程流程经验分享，没有配套公开仓库或可复现实现。
