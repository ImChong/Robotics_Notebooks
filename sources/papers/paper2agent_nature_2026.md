# Reimagining research papers as interactive and reliable AI agents（Nature）

> 来源归档（paper）

- **标题：** Reimagining research papers as interactive and reliable AI agents
- **作者：** Jiacheng Miao, Joe R. Davis, Yaohui Zhang, Jonathan K. Pritchard, James Zou 等
- **机构：** 斯坦福大学（Stanford University）
- **期刊：** Nature（2026-09-16）
- **DOI：** <https://doi.org/10.1038/s41586-026-11044-y>
- **论文：** <https://www.nature.com/articles/s41586-026-11044-y>
- **配套项目：** [Paper2Agent 项目页](../sites/paper2agent-ai.md) · [GitHub](../repos/paper2agent.md)
- **收录日期：** 2026-10-15

## 一句话摘要

Paper2Agent 将论文及其代码、数据和分析流程转为可供 AI agent 调用的 MCP 服务，并通过运行测试与对照原结果验证生成工具；跨领域案例显示它能把论文从静态说明转为可执行、可交互的研究入口。

## 核心摘录（面向 wiki 编译）

- **构建路径：** 分析论文与代码库，配置可运行环境，把方法封装为 MCP Tools、Resources 与 Prompts，再由测试 agent 运行示例、检查文件/数值/图像结果，反复修正后才发布通过验证的工具。
- **结果层级：** 若代码可运行，可提供执行型 MCP 工具；若代码或依赖不足，仍可提供论文与材料的结构化资源层，因此“agentified”不等于所有论文都能生成完整可执行方法。
- **规模评估：** 100 篇计算生物学论文中，74 篇成功生成 agent，提出 599 个工具，其中 593 个通过自动验证；300 个教程题上的准确率为 91.2±1.6%。另在 10 篇非生物计算论文的 42 个执行任务上，准确率为 98.1±0.8%。
- **使用边界：** 通过验证的工具可提高复现和调用便利性，但验证基于选定的参考代码、示例数据与容差，不能替代对新数据、前提假设或科学结论的独立审查。

## 对 wiki 的映射

- 升格实体页：[Paper2Agent](../../wiki/entities/paper-paper2agent.md) — 将论文与代码流程编译成经过运行验证的 MCP 工具和 paper skill。
- 交叉补强：[AI Auto-Research](../../wiki/concepts/ai-auto-research.md) — 研究工作流工具化和可复现性；[MCP](../../wiki/concepts/model-context-protocol.md) — 论文 agent 的调用接口。

## 参考来源（原始）

- Miao et al., Nature 2026：<https://www.nature.com/articles/s41586-026-11044-y>
- 补充资料与评测数据见 Nature 论文页面。
