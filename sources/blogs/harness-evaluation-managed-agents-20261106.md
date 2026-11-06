# 测试、语言模型评测与托管智能体：资料核验草稿

本记录为待完成 ingest 的资料清单，尚未升格为项目实体或完成源码分析。

## OpenJDK jtreg FAQ

- 原文：[jtreg FAQ](https://github.com/openjdk/jtreg/blob/master/src/share/doc/javatest/regtest/faq.md)，master 分支，blob `a02de1086a2d5d05ef0601ea695725aa0b809b76`。
- JavaTest 是运行测试程序的 harness；jtreg（旧称 regtest）提供识别和解释 JDK 测试描述的扩展。FAQ 将回归测试定义为针对缺陷修复并防止复发的测试，并介绍测试标签、运行选项与诊断。
- 待完成：检查官方可运行入口及对应源码，绘制经核实的流程图和源码时序图。

## 语言模型评测论文

- 原文：[Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782)；2024 年首次提交，当前 arXiv 页面列出 v3。
- 论文讨论语言模型评测的可复现性问题，作者结合 [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) 的开发经验提出建议。论文标题并非框架名。
- 待完成：取得并通读论文全文及必要附录，检查官方源码和运行入口，归档来源并绘制核实后的流程图及源码时序图。

## Anthropic Managed Agents

- 原文：[Scaling Managed Agents: Decoupling the brain from the hands](https://www.anthropic.com/engineering/managed-agents)，2026-04-08。
- 文章将 session 定义为追加式事件日志，将 harness 描述为模型调用及工具路由循环，将 sandbox 描述为执行环境；通过接口解耦这些组件，使其可独立替换或失败。
- 待完成：完整核对文章其余内容、公开实现状态和适用边界，建立实体及经核实的流程图。

## 验证状态

尚未完成全文与源码阅读、项目去重全面核验、本地 `make ci-preflight`，不得视为完成收录。
