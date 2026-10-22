# Harness² 官方项目页

> 官方项目页来源归档（核验日期：2026-10-22）。

- **URL：** <https://harness2.github.io/>
- **作者与单位：** Ruiyao Xu 等；Northwestern University、Google Cloud AI Research（据页面）
- **论文：** [Harness²: Recursive Agent Harnessing for an Open World（arXiv:2610.14224）](../papers/harness2_arxiv_2610_14224.md)
- **代码：** <https://github.com/google-research/harness2>
- **项目节点：** [Harness²](../../wiki/entities/paper-harness2.md)

## 项目页主张

Harness² 把 agent 表示为冻结的模型加可编辑的执行 harness，并进一步维护一个改进 harness：前者为当前任务调整，后者把可复用的 harness 与经验带入未来任务。

页面展示的任务级流程为参考执行 → invariant 与 domain-specific 两类提案 → 候选执行探测 → 合成后继 harness 并完成任务；跨任务层在任务完成后反思完整改进记录、更新经验库。后续任务从初始 harness 开始，按需复用经验。

## 可编辑组件

页面列出八类运行时文件：system prompt、guardrails、memory、tool guidance、subagents、skills、scripts、plugins。它们分别影响上下文构建、工具接口、任务编排与 middleware。

## 项目页展示的结果

- Gemini 3.5 Flash + OpenCode 在 WorkBuddy-Bench 平均分：base 72.4，单步 Harness² 77.5，parallel recursion（k=5）80.8。
- 页面还展示 JobBench 任务案例中的交付质量改进；该个案不等于 benchmark 总体平均值。

详细的多模型、多 benchmark 均值及标准差以[论文主表](../papers/harness2_arxiv_2610_14224.md)为准。

## 官方开放状态

页面提供论文、项目代码与可交互方法演示入口。仓库归档见[官方代码来源](../repos/google-research-harness2.md)；代码为 Apache-2.0，benchmark 与第三方依赖须分别遵循其上游许可。

## 原始出处

- 项目页：<https://harness2.github.io/>
- 论文：<https://arxiv.org/abs/2610.14224>
- 代码：<https://github.com/google-research/harness2>
