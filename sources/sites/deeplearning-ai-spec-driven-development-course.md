# Spec-Driven Development with Coding Agents（DeepLearning.AI 课程页）

- **标题：** Spec-Driven Development with Coding Agents
- **类型：** site / course-page
- **URL：** <https://www.deeplearning.ai/courses/spec-driven-development-with-coding-agents/>
- **合作方：** JetBrains
- **讲师：** Paul Everitt（JetBrains Developer Advocate）
- **配套仓库：** <https://github.com/https-deeplearning-ai/sc-spec-driven-development-files> — 归档见 [`sources/repos/sc-spec-driven-development-files.md`](../repos/sc-spec-driven-development-files.md)
- **社区材料帖：** <https://community.deeplearning.ai/t/course-materials/891543>
- **入库日期：** 2026-09-27

## 一句话摘要

短课：用 **Markdown 规格**（constitution + feature spec）替代 vibe coding，在 **plan → implement → verify** 环中让人类保持意图控制；演示 **AgentClinic** 全栈示例、**legacy 代码库 SDD 接入** 与可移植 **agent skill** 打包。

## 页面要点（截至入库日）

- **对比：** vibe coding 快但易偏离意图；SDD 用详细 spec 提升 **intent fidelity**、跨 session **保留上下文**、降低 **cognitive debt**。
- **你将学会：** 写 project constitution；feature spec 驱动实现与验证；在 greenfield / legacy 上复用同一工作流；把自定义流程封装为 **agent skill**（跨 agent / IDE 可移植）。
- **大纲：** 15 节（约 4m–6m 视频为主 + 1 篇 Setup 阅读）；**0 个站内 Code Examples**（代码在 GitHub 按视频文件夹提供）。
- **受众：** 已用过 LLM coding 工具、希望更 intentional 的开发者；需基础编程经验。
- **环境：** Node.js 18+、Git、coding agent（课内示例 **Claude Code**，强调 **agent-agnostic**）；IDE 示例 **WebStorm**。

## 开源核查（步骤 2.5，2026-09-27）

| 资产 | 状态 |
|------|------|
| 课程视频 | DeepLearning.AI 平台（需注册） |
| GitHub 伴学仓库 | **已开源** — 按 `VideoNN_*` 快照 + `prompts/` + `skills/` + `example_specs/` |
| 社区帖 | 讨论与材料索引（非代码仓） |

## 对 wiki 的映射

- [course-spec-driven-development-coding-agents](../../wiki/entities/course-spec-driven-development-coding-agents.md)
