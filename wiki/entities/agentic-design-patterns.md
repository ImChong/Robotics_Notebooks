---
type: entity
project_id: agentic-design-patterns
tags:
  - llm-agents
  - agent-design
  - multi-agent
  - rag
  - evaluation
  - safety
status: complete
updated: 2026-10-17
project: https://link.springer.com/book/10.1007/978-3-032-01402-3
code: https://github.com/evoiz/Agentic-Design-Patterns
related:
  - ../concepts/ai-agent-evaluation.md
  - ../concepts/agentic-coding-software-fundamentals.md
  - ../concepts/multimodality-basics.md
sources:
  - ../../sources/sites/agentic-design-patterns.md
  - ../../sources/repos/agentic-design-patterns.md
summary: "Antonio Gullí 的实践型智能体设计模式书，以 21 章和 7 个附录覆盖从提示链、路由、工具与规划到多智能体、RAG、安全及评测；可用于建立设计词汇与运行示例，但不是统一框架或比较基准，截图所示仓库的许可与官方关系仍需谨慎核验。"
---

# Agentic Design Patterns：智能体设计模式实践指南

## 一句话定义

Antonio Gullí 的 *Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems*（Springer Nature，2025）用模式分类和配套 notebook 介绍智能体系统的常见构造、运行机制与生产问题。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RAG | Retrieval-Augmented Generation | 书中作为知识检索模式，结合检索结果生成回答 |
| MCP | Model Context Protocol | 将外部工具或上下文接入 agent 的标准化接口主题 |
| A2A | Agent-to-Agent | 多智能体之间交换任务与结果的通信模式 |
| HITL | Human-in-the-Loop | 在自动流程中保留人工审阅或决策点 |
| GUI | Graphical User Interface | 附录讨论 agent 如何经由图形界面操作现实软件环境 |

## 为什么重要

- **提供一套可讨论的设计词汇：** 书将 agent 系统拆成路由、工具使用、规划、记忆、异常恢复、人工介入、检索、安全和监控等问题，便于团队把“做一个 agent”拆成可检查的设计选择。
- **例子可供上手：** GitHub 配套仓库按章节提供 notebook，多种框架实现有助于观察相似模式如何落在不同 SDK；它更像教程和模式目录，不是生产框架。
- **与机器人 agent 有交集：** 任务分解、工具调用、记忆、RAG、多 agent 协作、guardrail 与评测都可映射到机器人任务级规划及系统外围。它不覆盖机器人运动学、实时控制或安全控制器的设计。

## 核心结构

| 书中部分 | 主题 | 阅读时关注的问题 |
|----------|------|------------------|
| 核心模式（第 1–7 章） | 提示链、路由、并行化、反思、工具使用、规划、多智能体 | 任务何时应分解、分支、并行或交给其他 agent；如何控制工具边界 |
| 高级模式（第 8–11 章） | 记忆、学习与适应、MCP、目标设定与监控 | 状态如何跨步骤保存，工具接口如何稳定，目标如何持续检查 |
| 生产模式（第 12–14 章） | 异常恢复、人类参与、RAG | 失败如何恢复、哪些节点需要人审、检索如何约束生成 |
| 企业模式（第 15–21 章） | A2A、资源优化、推理、安全、评测监控、优先级、探索 | 多 agent 的通信与成本、安全边界和运行质量如何管理 |
| 附录（A–G） | 提示技巧、GUI 到现实环境、框架、AgentSpace、CLI、推理引擎、coding agents | 将模式映射到具体工具、交互界面和代码助手场景 |

模式之间应按任务约束组合，而不是全部叠加。例如，一个研究 agent 可先路由任务，再检索资料并调用工具，最后以独立 grader 检查输出；若关键操作会改变外部状态，则需在执行前加入确认与失败回滚。

## 工程实践

- **先写任务与终态，再选模式：** 明确输入、可观察状态、成功判据和失败代价；再判断是否需要拆分、路由、工具调用、记忆或人工介入。模式名本身不保证任务完成。
- **把 tool use 当作权限边界：** 为每个工具限制可执行动作、参数范围和凭证；高影响动作增加确认、幂等性、审计与回滚。提示词里的“请谨慎”不能替代运行时策略。
- **为每个模式建立评测：** 多轮任务记录 transcript 与环境 outcome，测试成功率、成本、延迟、失败恢复和安全违规。见 [AI Agent 评测](../concepts/ai-agent-evaluation.md)，不要只比较演示中的回答质量。
- **迁移到机器人时分层：** 把本书的模式应用在任务级编排、感知/规划模块选择和工具服务调用；运动控制与安全停机仍由经验证的控制和安全层负责。云端 notebook 不构成真机部署验证。
- **运行 notebook 前检查依赖：** 按单本示例固定 SDK、模型和 API 版本，隔离密钥，并记录服务成本与数据流向。仓库跨多个框架，不能假设存在一个统一的 `requirements.txt` 环境即可复现全部章节。

## 局限与风险

- **模式目录不等于经验定律：** 每种模式的适用性取决于任务分布、模型能力、工具可靠性和系统约束；书中示例不能替代目标场景的对照实验。
- **框架示例会漂移：** notebook 涉及外部 SDK 和模型接口，版本更新可能造成代码失效、行为变化或费用增加；复用前应锁版本并执行测试。
- **开放边界需谨慎表述：** Springer 页面标明书籍版权由作者所有并独占许可给 Springer Nature。截图所示 GitHub 仓库可公开访问，但 README 将书籍标为保留所有权利；其代码 MIT 声明指向的 `LICENSE` 文件在核查时不存在，且 GitHub 未识别仓库许可证。不要把 PDF 可下载等同于书籍或整仓采用开放许可。
- **来源关系未获出版社确认：** `evoiz/Agentic-Design-Patterns` 的 README 自称为全书材料仓并附作者链接，但出版社页面没有将其标注为官方 companion repo；本知识库将其描述为公开配套材料仓，不推断作者或出版方背书。
- **机器人安全与实时性不在覆盖范围：** 模式层的 guardrail、人工审批或多 agent 协商不能代替控制频率、碰撞约束、急停和硬件安全回路。

## 关联页面

- [AI Agent 评测](../concepts/ai-agent-evaluation.md) — 将模式放进可复现 task、grader 与环境终态的评测闭环。
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 书中评测章节落到具身 agent 时，按认知 / 世界模型 / 任务成功率 / sim↔real 四层选基准。
- [Agentic Coding 时代的软件工程基础](../concepts/agentic-coding-software-fundamentals.md) — 对照附录中的 coding agent 内容，理解工具之外仍需保留的架构、安全与运维判断。
- [多模态基础](../concepts/multimodality-basics.md) — 阅读书中 GUI 与现实环境 agent 示例的输入模态背景。

## 参考来源

- [Springer 图书页与版权信息](../../sources/sites/agentic-design-patterns.md)
- [GitHub 配套材料仓及开源范围核查](../../sources/repos/agentic-design-patterns.md)
- [出版社书目页](https://link.springer.com/book/10.1007/978-3-032-01402-3)

## 推荐继续阅读

- [Agentic Design Patterns — Springer Nature](https://link.springer.com/book/10.1007/978-3-032-01402-3) — 目录、章节信息与出版社版权说明。
- [evoiz/Agentic-Design-Patterns](https://github.com/evoiz/Agentic-Design-Patterns) — 可访问的 PDF 与章节 notebook；使用前核验版权和 notebook 依赖。
