# Movez：用 Gemini 4 Neon 搭建多 Agent 团队 / Gemini 4 Neon Agent Team in Eight Steps

- **类型：** blog
- **作者：** Movez（@0xMovez）
- **发布日期：** 2026-11-25
- **原文：** https://x.com/0xMovez/status/2125577359269019791
- **全文阅读版：** [Thread Navigator](https://threadnavigator.com/thread/2125577359269019791/)；[另一份阅读副本](https://www.xitter.gq/0xMovez/article/2125577359269019791)
- **入库日期：** 2026-11-28
- **核对范围：** X 原帖直连返回 403，按帖子 ID、署名与标题对照两份作者正文副本；通读导言、步骤 1–8、九条提示词的用途、计费说明和结语。交叉核对 Google 模型卡、SWE-sweep 评测方法、Antigravity 模型及 Swarm 文档、Gemini API 文件能力、T3 Code 集成 PR、Gemini API CLI v0.5.0 发布及 ADK 图工作流说明。
- **通读：** 已完成
- **代码开放状态：** 教程给出配置、提示词和 ADK 草图，未链接可直接运行整套方案的官方配套仓库或完整示例运行记录。T3 Code、Gemini API CLI 与 ADK 各自有独立上游代码，不等于教程实现、Neon 权重或 Antigravity 内部源码已开放。
- **沉淀到 wiki：** [Gemini 4 Neon 多 Agent 团队教程](../../wiki/entities/movez-gemini-4-neon-agent-team.md)

## 来源分工

- [Google DeepMind 模型卡](https://deepmind.google/models/model-cards/gemini-4-neon/)与[评测方法](https://deepmind.google/models/evals-methodology/gemini-4-neon)：核对 SWE-sweep 的测试设置和分数，不把跨 harness 的 12.8% 与 39.0% 写成流程中某一环的独立贡献。
- [Antigravity 模型](https://antigravity.google/docs/models)与 [Swarm](https://antigravity.google/docs/swarm/)：核对 Orchestrator 的双模型分工、Swarm 的共享板、工作区和模型默认值。教程中的社区实践是案例，不是平台默认行为或强制验收机制。
- [T3 Code PR #9348](https://github.com/pingdotgg/t3code/pull/9348)核对官方 ACP 可执行文件与认证方式；[Gemini API CLI v0.5.0](https://github.com/google-gemini/gemini-api-cli/releases/tag/v0.5.0)只作为教程对应的版本入口，不代表最新版或同一运行时。
- [Gemini API 文件切片说明](https://blog.google/innovation-and-ai/technology/developers-tools/file-slicing-controls/)核对函数返回中的文件引用与切片；[ADK 图工作流 Codelab](https://codelabs.developers.google.com/devsite/codelabs/build-agents-with-adk-graph-workflows)核对并行、聚合和条件回边。未执行厂商命令、付费模型或媒体渲染。
