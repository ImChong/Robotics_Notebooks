# Anthropic：Demystifying evals for AI agents

- **URL：** <https://www.anthropic.com/research/demystifying-evals-for-ai-agents>
- **类型：** 官方文档或作者工程文章
- **核验日期：** 2026-11-26
- **通读：** 已完成
- **核对范围：** 评测组成术语、Types of graders、Capability vs. regression evals、随机性与 pass@k / pass^k、收集评测任务小节。

## 来源摘要

文章把任务、重复尝试、过程轨迹、最终环境状态和评分器区分开，并列出程序、模型和人工评分的取舍。pass@k 关注多次尝试至少一次成功，pass^k 关注多次全部成功；两者不能脱离尝试数评估稳定性。

关联：[方法页](../../wiki/methods/ai-benchmark-evaluation.md)。
