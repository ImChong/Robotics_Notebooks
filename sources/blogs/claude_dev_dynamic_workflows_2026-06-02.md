# A harness for every task: dynamic workflows in Claude Code / 为每项任务动态生成专用 Harness

> 一手来源归档：Anthropic Claude Code 官方产品与工程博客的结构化研读记录；非逐字转载。

- **类型：** blog（agents）
- **作者：** Thariq Shihipar、Sid Bidasaria
- **机构：** Anthropic / Claude Code
- **发布日期：** 2026-06-02（以网页标注为准）
- **原文：** https://claude.dev/blog/a-harness-for-every-task-dynamic-workflows-in-claude-code/
- **版本：** 网页公开版，2026-11-19 核对
- **核对范围：** 通读 Example prompts、How dynamic workflows work、Why dynamic workflows、Dynamic vs static workflows、Helpful patterns、Use cases、When not to use、Tips、A new starting point 全部正文；逐项核对 API 概念、模式、案例及成本提醒。
- **通读：** 已完成
- **沉淀到 wiki：** [Claude Code 动态多 Agent 工作流](../../wiki/methods/claude-code-dynamic-workflows.md)
- **源码可用性：** 原文介绍产品的 JS workflow 构件并链接 Claude Code 官方文档，但未提供文章案例对应的独立公开可运行实现和固定源码版本；不伪造源码内部调用链。

## 核心机制

Claude Code 默认 harness 同时承担规划与执行，面对长时间、大规模并行、多级结构化或对抗性任务会遭遇「部分任务提前收尾」（agentic laziness）、自我偏好（self-preferential bias）与目标漂移（goal drift）。文章提出让 Claude 针对当前任务编写 JavaScript workflow，以不同上下文/职责的子 Agent 组合执行。

- `agent(prompt, opts)`：创建子 Agent；可选输出 schema、模型、隔离方式（worktree 或 remote）、agentType。
- `parallel()`：并行扇出并等待结果，再综合。
- `pipeline()`：让输入逐步经过流水线的各阶段。
- 标准 JavaScript 的 JSON、Math、Array 等负责确定性协调；脚本可基于任务需要选择模型与隔离策略。
- 文章称终端会话中断后恢复会话可续跑 workflow；可将 workflow 保存到 `~/.claude/workflows` 或作为 Skill 的文件复用。

## 六种流程模式

| 模式 | 用途 |
| --- | --- |
| Classify-and-act | 先分类任务，再选择 agent/执行路径 |
| Fan-out-and-synthesize | 并行切分小任务，等待全部完成后结构化汇总 |
| Adversarial verification | 独立审查者根据 rubric 质疑执行者结论 |
| Generate-and-filter | 生成候选、去重后按标准过滤 |
| Tournament | 让多个候选逐轮成对比较，适合主观排序 |
| Loop until done | 在未知工作量时反复运行，直到明确终止条件 |

## 典型场景与边界

文章列举跨文件重构、深度研究、逐 claim 复核、上千项候选排序、反复挖掘团队规则、对立假设根因调查、工单 triage、模型路由与 eval；其中工单 triage 用「只读隔离区」处理不可信输入，再让有权限 actor 接收已筛选的摘要执行动作。复核场景可使用每条事实一个独立 checker 和 source auditor。

**什么时候不用：** 简单编码任务未必需要 workflow；多 Agent 协调和独立评审会增加 token 消耗，单 Agent 可完成时不该只为并行而并行。文章建议可明确 token 预算；重复任务可和 `/loop`、`/goal` 配合，但要有可执行终止标准。原文没有可用于普遍定量对比的公开基准成绩。

## 关联资料

- [Claude Docs chaos loop 官方原文归档](claude_dev_chaos_loop_docs_launch_2026-11-17.md)
- [多 Agent 编码管理与认知把关](../../wiki/concepts/coding-agent-orchestration-and-cognitive-oversight.md)
- [Coding Agent 验证闭环](../../wiki/methods/coding-agent-verification-loops.md)

## 原文入口

- [A harness for every task: dynamic workflows in Claude Code（2026-06-02）](https://claude.dev/blog/a-harness-for-every-task-dynamic-workflows-in-claude-code/)
