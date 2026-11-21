# Harness engineering: leveraging Codex in an agent-first world

> 来源归档；OpenAI 官方工程文章的结构化摘录。

- **类型：** blog（OpenAI 官方 Engineering 博客）
- **作者：** Ryan Lopopolo（Member of the Technical Staff）
- **发布日期：** 2026-02-11
- **原文：** https://openai.com/index/harness-engineering/
- **入库日期：** 2026-11-21
- **核对范围：** 全文各节：empty repo、engineer role、application legibility、repo knowledge、agent legibility、architecture enforcement、throughput/merge、agent-generated、autonomy、entropy/garbage collection、unknowns。
- **通读：** 已完成
- **开源状态：** 介绍内部试验产品及工程体系；文章未提供此产品的公开可运行仓库，不能把其内部工具链当开源系统。
- **沉淀到 wiki：** [Agent-first 软件工厂与 Harness Engineering](../../wiki/concepts/agentic-software-factory-harness.md)

## 核心结论

文章记录从空仓库开始、由 Codex 生成全部应用代码、测试、CI、文档与可观测设施的内部产品工程试验。约五个月内 3 名工程师推动约 1,500 个已合并 PR、约百万行代码；作者估算相对人工编写用时约 1/10。这些数字来自内部案例及作者估计，不是独立实验测量。

**Humans steer; agents execute.** 人负责定义目标、构造环境、提供反馈和验收；系统不足时优先补工具、文档、测试与能力，而非仅重新提示模型“努力”。

## Harness 的关键部件

1. **Agent 可读的环境：** 每个 git worktree 单独启动应用；通过 Chrome DevTools Protocol 读取 DOM、截图、操作浏览器；日志、指标、链路追踪在隔离可观测环境中可查询（文中举 LogQL、PromQL）。
2. **仓库是系统记录：** AGENTS.md 是短地图，而非千页手册；详细设计文档、执行计划、产品规格、质量与安全约束放入可版本化的 docs/，以渐进加载获取上下文。
3. **机械约束：** lint/CI 强制架构规则、依赖方向及文档新鲜度；允许代理自主迭代，但自动检查结果应可复验。
4. **持续清理：** 定期 doc-gardening 和 golden principles 检查漂移、技术债与重复模式，自动发起小型 PR；避免把生成式重复造成的坏模式不断放大。
5. **端到端闭环：** 代理复现 bug → 视频证据 → 修复 → 操作应用验证 → 录制修复证明 → PR/反馈/构建纠错 → 必要时向人升级 → 合并。

## 适用边界

- 文中较宽松的 merge gate 针对特定高吞吐、可快速纠错的内部产品，**不是**通用的安全部署规范。
- 作者明确不知道完全由 agent 生成的系统能否在多年尺度保持架构一致性。
- 机器人对应实现可借鉴任务级工具、仿真 QA、结构化日志与回滚，但真机动作安全、急停和控制实时性不能从这篇软件经验推导出来。

## 官方来源

- [OpenAI, Harness engineering: leveraging Codex in an agent-first world（2026-02-11）](https://openai.com/index/harness-engineering/)
