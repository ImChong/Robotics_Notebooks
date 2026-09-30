---
type: entity
tags:
  - llm-agents
  - coding-agents
  - skills
  - productivity
  - agent-infrastructure
status: complete
updated: 2026-09-30
related:
  - ./mattpocock-skills.md
  - ./humanlayer-skills.md
  - ./hermes-agent.md
  - ../references/llm-wiki-karpathy.md
sources:
  - ../../sources/repos/mattpocock-skills.md
  - ../../sources/sites/skills-sh-mattpocock-selected-skills.md
summary: "handoff 把当前会话压缩为 OS 临时目录下的交接文档，引用已有 spec/ADR/issue 路径，并建议下一 agent 应调用的 skills；适合 Cloud Agent 换会话续作。"
---

# handoff（Matt Pocock Skill）

**handoff**（[skills.sh](https://skills.sh/mattpocock/skills/handoff)）生成 **跨会话交接文档**：摘要当前进度、**不重复** 已有 artifact（spec、ADR、issue、commit），并列出 **suggested skills**；输出写到 **用户 OS 临时目录**（非 workspace），且 **脱敏** API key/PII。

## 一句话定义

**会话压缩 + 指针化引用** — 让新 agent 从 **路径与 skill 名** 续作，而非重读整段 chat。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ADR | Architecture Decision Record | 交接文引用而非复制 |
| PII | Personally Identifiable Information | 必须 redact |
| LLM | Large Language Model | 写 handoff 的 agent |

## 为什么重要（对本知识库读者）

- **Cloud Agent 多轮：** Cursor Cloud 任务常 **换 run 续作**；handoff 与 [HumanLayer Skills](humanlayer-skills.md) 的 **持久 loop** 互补 — 前者 **adhoc 会话**，后者 **定时 PR 维护**。
- **Wiki 维护：** 长 ingest 后 handoff 应指向 **`make ci-preflight` 状态、分支名、待测 page-id**，而非粘贴整篇 wiki diff。

## 核心规则

- 可选 `argument-hint`：用户描述 **下一 session 焦点**
- **suggested skills** 段：明示下一 agent 应 `Skill tool` 的名称
- 敏感信息 redact

## 常见误区或局限

- **误区：替代 git commit。** 交接文 **不** 代替版本化记录。
- **局限：** 临时目录路径因 OS 而异；需用户在下一 session 提供文件。

## 关联页面

- [mattpocock/skills 总览](mattpocock-skills.md)
- [HumanLayer Skills](humanlayer-skills.md)
- [Hermes Agent](hermes-agent.md)

## 参考来源

- [handoff SKILL.md](https://github.com/mattpocock/skills/tree/main/skills/productivity/handoff)
- [mattpocock/skills 归档](../../sources/repos/mattpocock-skills.md)

## 推荐继续阅读

- [skills.sh handoff](https://skills.sh/mattpocock/skills/handoff)
