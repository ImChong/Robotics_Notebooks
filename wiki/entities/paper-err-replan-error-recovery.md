---
type: entity
tags:
  - paper
  - vlm
  - embodied-ai
  - error-recovery
  - reinforcement-learning
  - benchmark
status: complete
updated: 2026-10-22
project_id: err-replan-error-recovery
arxiv: "2610.27003"
related:
  - ../overview/hub-embodied-foundation-model.md
  - ../overview/hub-embodied-eval-benchmark.md
  - ../concepts/ai-agent-evaluation.md
sources:
  - ../../sources/papers/err_replan_error_recovery_arxiv_2610_27003.md
summary: "ERR（2026）：用 ERR-Env/ERR-Bench 评测具身 VLM 的错误恢复，并以 AGT-GSPO（turn 级奖励 + action-grouped 优势）训练 2B 的 ERR-VLM，提升严格成功率与恢复率。"
---

# ERR: Replan for Error Recovery（具身 VLM 在线错误恢复）

## 一句话定义

**ERR 把「执行失败后能否重新规划并恢复」变成可验证的交互环境与基准，并用 turn 级 RL 训练具身 VLM 完成恢复。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 具身决策的视觉语言模型 |
| ERR | Error Recovery / Replan | 本文任务：错误恢复式重规划 |
| AGT-GSPO | Action-Grouped Turn-GSPO | 本文的在线 RL 方法；GSPO 为组序列策略优化类方法（据名称推测，摘要未展开） |
| SSR | Strict Success Rate | 严格成功率，正文指标 |
| RR | Recovery Rate | 恢复率，正文指标 |

## 为什么重要

- 具身 VLM 的单步执行出错后，环境可能进入需要重新规划的状态；只评测任务进度会掩盖执行可靠性问题。
- 摘要指出不同 VLM 之间，任务进度与执行可靠性并不一致。

## 方法与机制

- **ERR-Env：** 给出任务进度与错误反馈的交互环境。
- **AGT-GSPO：** 在线 RL，使用 turn 级奖励和 action-grouped 优势估计。
- **ERR-VLM：** 2B 紧凑模型，由该方法训练。

## 实验与评测

- 基准：ERR-Bench，衡量恢复能力与执行可靠性。
- AGT-GSPO 使 ERR-VLM 的 SSR 提升 10.25 点，RR 提升 22.95 点（据摘要）。

## 与其他工作对比

- 摘要未给出与其他 VLM 的完整数值对比；仅说明任务进度与执行可靠性在不同模型间不一致。

## 结论

本文的主要贡献是把错误恢复变成可训练、可验证的目标。

- 评测具身 VLM 时，应同时报告任务进度与执行可靠性，二者可能不一致。
- turn 级奖励使 RL 能针对恢复步骤，而不只针对最终结果。
- 证据目前为摘要级，具体环境、奖励设计与对比基线需读全文确认。

## 工程实践

- 代码与项目页：抓取页面未列出。
- 复现前需要：ERR-Env 与 ERR-Bench 的获取方式（未核实）。

## 局限与风险

- 2B 模型在摘要中的规模较小，能否迁移到更大模型未经验证。
- 推测：基于 VLM 的错误判断可能引入评估噪声，需要检查奖励是否被 VLM 评判偏差影响。

## 关联页面

- [具身大模型分类学选型闭环](../overview/hub-embodied-foundation-model.md) — 具身 VLM 的模型选型背景
- [具身评测基准选型闭环](../overview/hub-embodied-eval-benchmark.md) — 同类的执行可靠性评测问题

## 参考来源

- [ERR 论文来源归档](../../sources/papers/err_replan_error_recovery_arxiv_2610_27003.md)

## 推荐继续阅读

- [arXiv 2610.27003](https://arxiv.org/abs/2610.27003)
