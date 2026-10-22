---
type: entity
tags:
  - paper
  - multi-agent
  - coding-agents
  - agent-harness
  - microsoft
status: complete
updated: 2026-10-22
project_id: agensh
arxiv: "2609.26781"
project: https://agens-harness.github.io/project/
related:
  - ../entities/rle-bench.md
  - ../concepts/ai-agent-evaluation.md
  - ../overview/hub-embodied-eval-benchmark.md
sources:
  - ../../sources/papers/agensh_arxiv_2609_26781.md
summary: "Agensh（2026，Microsoft Research）：无中心 orchestrator 的多 agent harness，通过共享工作区、消息与共享上下文协调，最多 1,024 个 agent；pandoc 通过率 33.89% → 55.06%。"
---

# Agensh（多 agent 组织智能扩展）

## 一句话定义

**Agensh 是一个没有中心编排者的多 agent harness：worker 通过共享工作区、消息与共享上下文自行认领、执行与合并，并把规模扩大到 1,024 个 agent。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| harness | Agent harness | 管理多个 agent 协作的运行框架（本文主要贡献） |
| MA | Multi-Agent | 多 agent 协作 |
| ProgramBench | ProgramBench | 论文使用的编程任务基准；摘要未展开细节 |
| pass rate | Test pass rate | 测试通过率，本文的主要结果指标 |

## 为什么重要

- 多 agent 系统常依赖中心 orchestrator，规模变大时成为瓶颈；本文探讨无中心协调能否随规模提升效果。
- 对具身与机器人软件工程而言，coding agent 协作（如 [RLE-Bench](./rle-bench.md) 所评测的工程闭环）是潜在应用方向（推测，论文未针对机器人任务）。

## 方法与机制

- 无中心 orchestrator：worker 异步协调，包括收集上下文、认领与分配子任务、执行、共享发现、验证与合并。
- 三个组件：共享工作区（追踪已提出、进行中、已完成的工作）、worker 间消息接口、共享上下文（存放可复用发现与工作意图）。

## 实验与评测

- ProgramBench 最难的 5 个任务（GPT-5.6-sol high）：1 → 128 个 agent，平均测试通过率 19.31% → 28.78%（约相对提升 49%）。
- pandoc 从零构建：1 → 1,024 个 agent，通过率 33.89% → 55.06%。

## 与其他工作对比

- 摘要称规模更大的组织更快达到相近通过率，且协作模式随规模出现并趋于标准化。
- 摘要未给出与中心 orchestrator 方案的直接数值对比。

## 结论

本文的主要证据是：在所测任务上，去中心化 agent 组织的规模扩展有正向效果。

- 去中心的协调可以不依赖中心 orchestrator 而随规模提升结果。
- 共享工作区与共享上下文是协调的关键组件（据摘要的组件划分）。
- 证据来自两个编程任务，尚未在机器人任务上验证。

## 工程实践

- 代码：`microsoft/Agensh` 直接抓取返回 404，仓库可能未公开或尚未创建（未核实）。
- 项目页含回放 demo，可用于理解协调过程。

## 局限与风险

- 结果依赖特定模型（GPT-5.6-sol high）与基准，跨模型结论未经摘要验证。
- 1,024 个 agent 的成本与可复现性未在摘要中讨论。

## 关联页面

- [RLE-Bench](./rle-bench.md) — 同样面向 coding agent 的工程能力评测
- [AI Agent 评估](../concepts/ai-agent-evaluation.md) — agent 结果评估的背景

## 参考来源

- [Agensh 论文来源归档](../../sources/papers/agensh_arxiv_2609_26781.md)

## 推荐继续阅读

- [arXiv 2609.26781](https://arxiv.org/abs/2609.26781)
- [项目页](https://agens-harness.github.io/project/)
