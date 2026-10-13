---
type: concept
tags: [software-engineering, specification-driven-development, coding-agents, requirements-engineering]
status: complete
updated: 2026-10-13
related:
  - ./test-driven-development.md
  - ./behavior-and-acceptance-driven-development.md
  - ../comparisons/sdd-tdd-bdd-development-methods.md
  - ../entities/course-spec-driven-development-coding-agents.md
sources:
  - ../../sources/papers/spec-driven-development-arxiv-2602-00180.md
  - ../../sources/repos/github-spec-kit.md
summary: "规格驱动开发（SDD）把可维护的需求与验收规格置于实现之前或实现过程中，持续追踪规格、计划、任务、代码和验证；它是外层需求治理方式，可与 TDD / BDD 组合。"
---

# 规格驱动开发（SDD）

**规格驱动开发（Specification-Driven Development, SDD）** 是将可审阅、可追溯的规格作为需求理解与实现之间的持久契约，并用规格引导计划、代码生成/修改和验证的开发方法；在 AI 编码场景下，它特别用于让 agent 在跨会话和多步执行中保留任务意图与验收边界。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDD | Specification-Driven Development / Spec-Driven Development | 以规格组织实现与验证的开发方法；命名略有差异 |
| SDLC | Software Development Life Cycle | 需求、设计、实现、测试、交付等软件生命周期 |
| FR | Functional Requirement | 系统需提供的行为或能力 |
| NFR | Non-Functional Requirement | 性能、安全、可用性等质量约束 |
| CLI | Command-Line Interface | Spec Kit 等工具常用的命令行入口 |

## 核心想法：规格成为可追溯的工作对象

SDD 不只是“先写一份需求文档”。规格需要在实现过程中保持可用：每一项需求都应有对应的设计决策、任务或验证证据；代码或需求改变时，相关规格也要更新。它可以由人实现，也可以交给 coding agent 执行，不能把“生成了代码”直接当成“满足规格”。

当下 SDD 的用法并不完全统一。一篇 AI 编码相关预印本将其分成 **spec-first → spec-anchored → spec-as-source** 的规格权威度连续谱：规格可以只是实现起点，也可以和代码互相校验，或成为代码生成/持续验证的权威来源。这个分类是作者提出的实践框架，不是行业标准。

## 一种可执行工作流

GitHub Spec Kit 是这一理念的开源工具实现之一；其流程可概括为：

1. **Constitution：** 写清产品目标、硬约束、默认技术栈和质量原则。
2. **Specify / Clarify：** 把用户目标整理为可验证的行为需求、边界情况和不做事项；用澄清问题消除歧义。
3. **Plan：** 把需求映射到架构、数据模型、接口、测试策略与依赖。
4. **Tasks：** 将计划拆成小而可审阅的任务，并标明对应的需求/验收条件。
5. **Analyze：** 检查需求、计划、任务间是否存在冲突、遗漏或失去追溯关系。
6. **Implement / Verify：** 增量实现；运行自动化测试、静态检查及必要的人工/仿真验收。
7. **Converge：** 需求演化时更新规格和计划，再复查实现是否仍满足当前规格。

### 流程总览

```mermaid
flowchart LR
  goal["用户目标与约束"] --> spec["可验证规格"]
  spec --> clarify["澄清边界与示例"]
  clarify --> plan["技术方案与验证策略"]
  plan --> tasks["可追溯任务"]
  tasks --> impl["增量实现"]
  impl --> verify["自动化与人工验证"]
  verify -->|不满足或需求改变| spec
  verify -->|满足当前规格| ship["审阅与交付"]
  ship -->|新需求| spec
```

## 规格写到什么程度

优先写清**能改变实现或验收结论**的信息，而不是追求文档厚度：

| 规格项 | 好的表达 |
|---|---|
| 行为 | 输入、触发条件、输出状态；避免只写“体验更好” |
| 边界 | 空值、超界、超时、并发、错误恢复与权限 |
| 约束 | 延迟、吞吐、资源、平台、兼容性和安全不变量 |
| 验收 | 可观察的成功/失败条件，以及数据、环境与测量口径 |
| 非目标 | 本次明确不做的能力，避免 agent 自行扩展范围 |
| 来源链接 | 每条需求为何存在、由谁确认、对应哪项测试或证据 |

例如机器人任务不能只写“G1 能推椅子”：还需约束目标区域、碰撞/接触边界、允许的底盘与手臂动作、完成判据、失败恢复和仿真/真机的验证范围。

## 工程实践与局限

| 适合 | 风险 / 控制方式 |
|---|---|
| 多步骤、跨人/agent、需审计或返工成本高的功能 | 规格可能过厚；仅把影响代码与验收的约束写入主规格 |
| 对安全、接口、非功能约束敏感的系统 | 规格写得像愿望清单时不可验证；给出阈值、场景和测量口径 |
| AI agent 执行多个任务或跨会话工作 | agent 可能误读或漏做；对关键假设明确要求人确认，并独立验证结果 |
| 需求频繁演进的产品 | 文档会漂移；把规格演化纳入任务、评审与 CI 的变更路径 |
| 已有复杂代码库 | 可能无法一次性补齐完整规格；从高风险边界或当前变更切片开始 |

SDD 提供的是**需求—计划—实现—证据的外层可追溯性**，不自动创造好的需求，也不替代测试。将测试先写并快速反馈的内环见 [TDD](./test-driven-development.md)；跨角色通过具体行为例子达成共识见 [BDD / ATDD](./behavior-and-acceptance-driven-development.md)。

## 结论

**SDD 的价值不在“多写文档”，而在让重要需求、实现任务和验证证据保持可追溯；是否值得采用，取决于遗漏风险是否高于维护规格的成本。**

1. 先为高风险、高协作成本的切片建规格，不必对每个小改动都启动完整 SDD。
2. 每条关键需求都应连到任务和可执行或可审阅的验收证据。
3. 需求变化后同步改规格；旧规格不能因“已经写过”而继续充当真相。
4. 对 agent 工作显式写非目标、边界和验证命令，不能只写自然语言愿景。
5. 将 SDD 与 TDD/BDD 组合：规格回答“做什么、为何、怎么判定”，内层测试循环回答“怎样小步实现并获得反馈”。

## 关联页面

- [TDD：测试驱动开发](./test-driven-development.md)
- [BDD 与 ATDD：行为及验收测试驱动开发](./behavior-and-acceptance-driven-development.md)
- [SDD / TDD / BDD / ATDD 对照](../comparisons/sdd-tdd-bdd-development-methods.md)
- [Spec-Driven Development with Coding Agents 短课](../entities/course-spec-driven-development-coding-agents.md)

## 参考来源

- [Spec-Driven Development 预印本摘录](../../sources/papers/spec-driven-development-arxiv-2602-00180.md)
- [GitHub Spec Kit 项目归档](../../sources/repos/github-spec-kit.md)

## 推荐继续阅读

- [GitHub Spec Kit 官方方法与工作流](https://github.github.io/spec-kit/)
- [Spec-Driven Development: From Code to Contract](https://arxiv.org/abs/2602.00180)
