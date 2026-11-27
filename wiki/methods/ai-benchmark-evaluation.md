---
type: method
tags: [evaluation, benchmark, ai]
status: complete
updated: 2026-11-27
summary: "使用明确的数据、任务协议和指标比较模型或智能体的行为。"
sources:
  - ../../sources/sites/method-ai-benchmark-evaluation-primary-20261126.md
  - ../../sources/blogs/anthropic_demystifying_evals_ai_agents_2026-01-09.md
original:
  - ../../sources/sites/method-ai-benchmark-evaluation-primary-20261126.md
related:
  - ./embodied-agent-benchmarking.md
  - ../overview/hub-embodied-eval-benchmark.md
  - ../concepts/ai-agent-evaluation.md
---

# AI Model and Agent Benchmark Evaluation · AI 模型与智能体基准评测

## 一句话定义

预先规定输入、运行约束和判分方式，用可重放数据度量模型或智能体表现。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| AI | Artificial Intelligence | 鉴别评测对象是模型还是完整系统 |
| LLM | Large Language Model | 可按量表辅助开放输出评分 |
| CI | Continuous Integration | 自动运行固定回归集 |

## 为什么重要

同一任务可以得到正确文本、错误状态或正确状态但违规路径；只看一个结果或自报成功会掩盖问题。区分能力测试与回归测试，避免把低样本随机波动当成持续进步。

[Gemini 蒸馏服务](../entities/google-gemini-distillation-service.md) 是托管文本调优的例子：用同一留出集比较基座学生、蒸馏学生和教师，并将质量、耗时、成本分别汇报。

## 主要技术路线（核心机制）

固定模型、工具集、外部环境和调用预算，把资源限制也视为可比较条件。程序验证可测工件和最终状态，开放质量再用明确量表和人类抽查；同一任务重复尝试不跨 trial 继承状态。

```mermaid
flowchart LR
  N0["固定任务和验收口径"]
  N1["独立重置后多次运行"]
  N2["保存轨迹和最终状态"]
  N3["程序判分与人工抽查"]
  N4["分任务聚合并报告预算"]
  N0 --> N1
  N1 --> N2
  N2 --> N3
  N3 --> N4
```

## 工程实践

例如生成机器人数据质检报告：要求缺帧数量与已知标签比对、结果工件可读；将正常与异常 run 均纳入。报告通过率及分母、失败分类和耗时；若使用 pass@k 或 pass^k，写明 k 和 trial 采样方式。

[Mood Recognition 的网页交互评测](../entities/libraries-dev-mood-recognition.md) 是分母不能混用的例子：分别汇报预设障碍的检出次数与平静时间的误标占比，两者不能合并成一个准确率。

## 局限与风险

训练集污染、任务覆盖偏窄和评分规则错位都可抬高分数。榜单值不是机器人部署验收；具身任务还需要控制相机、本体、动作接口和场景。

## 关联页面

- [具身智能体评测](embodied-agent-benchmarking.md)
- [具身评测与基准导航](../overview/hub-embodied-eval-benchmark.md)
- [AI Agent 评测](../concepts/ai-agent-evaluation.md)

## 参考来源

- [Anthropic：Demystifying evals for AI agents 核读归档](../../sources/sites/method-ai-benchmark-evaluation-primary-20261126.md)
- [原页面关联资料](../../sources/blogs/anthropic_demystifying_evals_ai_agents_2026-01-09.md)

## 推荐继续阅读

- [Anthropic：Demystifying evals for AI agents](https://www.anthropic.com/research/demystifying-evals-for-ai-agents)
