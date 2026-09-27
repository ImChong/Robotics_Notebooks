# Spec-Driven Development with Coding Agents（DeepLearning.AI 短课）

> 来源归档（ingest）

- **标题：** Spec-Driven Development with Coding Agents
- **类型：** course
- **课程页：** <https://www.deeplearning.ai/courses/spec-driven-development-with-coding-agents/>
- **机构 / 合作：** DeepLearning.AI × **JetBrains**；讲师 **Paul Everitt**
- **伴学仓库：** <https://github.com/https-deeplearning-ai/sc-spec-driven-development-files> — [`sources/repos/sc-spec-driven-development-files.md`](../repos/sc-spec-driven-development-files.md)
- **社区：** <https://community.deeplearning.ai/t/course-materials/891543>
- **入库日期：** 2026-09-27
- **一句话说明：** 用 constitution 与 feature spec 驱动 coding agent，plan-implement-verify 闭环，并封装为可移植 agent skills。

## 核心摘录（面向 wiki 编译）

- **SDD vs vibe coding：** 先写清 **what to build**（Markdown spec），再让 agent 实现；强调 **human-in-the-loop** 与可维护性。
- **Project constitution：** 与 agent 协作产出 `mission.md`、`tech-stack.md`、`roadmap.md`。
- **Feature 三件套：** `plan.md`、`requirements.md`、`validation.md` → 实现 → 按 validation 验收。
- **Replan / MVP：** 功能间重规划；两阶段 feature 后收敛 MVP。
- **Legacy：** 用现有文档生成 spec，把 SDD 引入旧代码库（课内 Feedback Form 等示例）。
- **Skills：** 将自定义工作流打包为 **agent skill**（课内 feature-spec、changelog），强调 **agent replaceability**。
- **示例栈：** AgentClinic（Node 18+，TypeScript，Hono）；agent 示例 Claude Code，流程声称 **agent-agnostic**。
- **对 wiki 的映射：** [course-spec-driven-development-coding-agents](../../wiki/entities/course-spec-driven-development-coding-agents.md)

## 课程大纲（15 节，页面列出）

Introduction → Why SDD → Workflow overview → Set up (reading) → Setup → Creating the constitution → Feature specification → Implementation → Validation → Project replanning → Second feature → MVP → Legacy support → Build your own workflow → Agent replaceability → Conclusion

## 当前提炼状态

- [x] 课程页 + GitHub README 交叉核查
- [x] wiki 实体新建
