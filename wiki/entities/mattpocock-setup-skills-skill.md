---
type: entity
tags:
  - llm-agents
  - coding-agents
  - skills
  - software-engineering
  - agent-infrastructure
status: complete
updated: 2026-09-30
related:
  - ./mattpocock-skills.md
  - ./mattpocock-triage-skill.md
  - ./mattpocock-grill-with-docs-skill.md
  - ../../AGENTS.md
  - ../../schema/ingest-workflow.md
sources:
  - ../../sources/repos/mattpocock-skills.md
  - ../../sources/sites/skills-sh-mattpocock-selected-skills.md
summary: "setup-matt-pocock-skills 是每个仓库一次性 bootstrap：配置 issue tracker（GitHub/GitLab/本地 .scratch）、triage 五标签词表、GLOSSARY/ADR 布局，并写入 CLAUDE.md 或 AGENTS.md 的 Agent skills 段。"
---

# setup-matt-pocock-skills（Matt Pocock Skill）

**setup-matt-pocock-skills**（[skills.sh](https://skills.sh/mattpocock/skills/setup-matt-pocock-skills)）是 mattpocock 工程技能包的 **前置安装**：探索仓库现状后，与用户 **分段确认** issue 来源、triage 标签、单/多 context  glossary 布局，并写入 `docs/agents/*.md` 与根级 **Agent skills** 段。

## 一句话定义

**一次 setup，统一假设** — 让 triage、to-issues、grill-with-docs、tdd 等技能读到 **同一套 tracker 与词汇路径**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ADR | Architecture Decision Record | 默认 `docs/adr/` |
| CLI | Command-Line Interface | GitHub `gh`、GitLab `glab` |
| PR | Pull Request | 可选把外部 PR 纳入 triage 面 |

## 为什么重要（对本知识库读者）

- **本仓库 AGENTS.md：** Cloud Agent 已含 Cursor 专用说明；若在 **下游应用仓** 安装 mattpocock 技能，setup 决定写 **CLAUDE.md 还是 AGENTS.md**（与 [本站 AGENTS.md](../../AGENTS.md) 角色类似）。
- **Robotics_Notebooks 本身：** 主维护流是 **issue/PR + ingest**；triage 技能更适 **产品应用仓**，但 setup 文档化 **tracker 约定** 仍有借鉴价值。

## 核心产出

| 文件 | 内容 |
|------|------|
| `docs/agents/issue-tracker.md` | GitHub / GitLab / local / 自由文本 |
| `docs/agents/triage-labels.md` | 五角色标签（若安装 triage） |
| `docs/agents/domain.md` | GLOSSARY / ADR / monorepo map |
| `CLAUDE.md` 或 `AGENTS.md` | `## Agent skills` 块 |

## 常见误区或局限

- **误区：可跳过。** README 要求安装 mattpocock 包后 **必须** 跑 setup。
- **局限：** 非 GitHub 的 Linear/Jira 仅 **自由文本** 记录，自动化程度取决于 harness。

## 关联页面

- [mattpocock/skills 总览](mattpocock-skills.md)
- [triage](mattpocock-triage-skill.md)
- [Ingest Workflow](../../schema/ingest-workflow.md)

## 参考来源

- [setup SKILL.md](https://github.com/mattpocock/skills/tree/main/skills/engineering/setup-matt-pocock-skills)
- [mattpocock/skills 归档](../../sources/repos/mattpocock-skills.md)

## 推荐继续阅读

- [skills.sh setup-matt-pocock-skills](https://skills.sh/mattpocock/skills/setup-matt-pocock-skills)
