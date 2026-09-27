# sc-spec-driven-development-files（DeepLearning.AI × JetBrains 伴学仓）

- **标题：** Spec-Driven Development with Agentic Coding Assistants（伴学代码）
- **类型：** repo / course-materials
- **链接：** <https://github.com/https-deeplearning-ai/sc-spec-driven-development-files>
- **课程页：** <https://www.deeplearning.ai/courses/spec-driven-development-with-coding-agents/>
- **入库日期：** 2026-09-27
- **一句话说明：** AgentClinic 示例项目按视频分文件夹快照；含 constitution/spec 模板、逐视频 prompts、changelog 与 feature-spec **agent skills**。

## 为什么值得保留

- **可复现 SDD 工作流：** 从空 scaffold → constitution → Phase 1/2 feature → MVP → legacy 重建 → 自定义 skill，每步有 **起始树 + prompts.md**。
- **与机器人侧同构：** 真机/仿真 [autoresearch harness](../../wiki/queries/real-robot-policy-autoresearch-harness.md) 也依赖 **written spec + verify**；本仓是通用软件侧的教科书式模板。

## 目录结构（README 摘要）

| 路径 | 用途 |
|------|------|
| `Video05_*` … `Video14_*` | 各视频 **起始** 完整项目态（AgentClinic，TypeScript / Hono） |
| `prompts/` | 全课程编号 prompt 合集 |
| `skills/` | 可复用 agent skills（如 changelog、feature-spec） |
| `example_specs/` | 规格文档范例 |

**建议跟做：** 从 Video 5 起顺序构建；或 `cp -r VideoNN_* my-agentclinic && npm install` 跳章。

## 开源状态

- **已开源**（GitHub 公开；API 未返回 SPDX license 字段，以仓库为准）。

## 对 wiki 的映射

- [course-spec-driven-development-coding-agents](../../wiki/entities/course-spec-driven-development-coding-agents.md)
- [Agentic Coding 软件工程基础](../../wiki/concepts/agentic-coding-software-fundamentals.md)
