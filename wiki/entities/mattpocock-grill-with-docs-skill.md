---
type: entity
tags:
  - llm-agents
  - coding-agents
  - skills
  - software-engineering
  - domain-driven-design
  - agent-infrastructure
status: complete
updated: 2026-09-30
related:
  - ./mattpocock-skills.md
  - ./mattpocock-grill-me-skill.md
  - ./mattpocock-setup-skills-skill.md
  - ../references/llm-wiki-karpathy.md
  - ../../schema/ingest-workflow.md
sources:
  - ../../sources/repos/mattpocock-skills.md
  - ../../sources/sites/skills-sh-mattpocock-selected-skills.md
summary: "grill-with-docs 在 grilling 对齐同时调用 domain-modeling，边问边更新 GLOSSARY.md 与 docs/adr/，把 ubiquitous language 与架构决策写入仓库文件。"
---

# grill-with-docs（Matt Pocock Skill）

**grill-with-docs**（[skills.sh](https://skills.sh/mattpocock/skills/grill-with-docs)）在 [grill-me](mattpocock-grill-me-skill.md) 的 **grilling** 之上叠加 **`domain-modeling`**：对齐过程中 **即时** 更新 `GLOSSARY.md` 与 ADR，减少长 chat 重复解释术语。

## 一句话定义

**对齐 + 文档化**：同一轮 grilling 里把领域词与架构决策 **写进 repo**，服务后续 TDD/triage/架构 review。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ADR | Architecture Decision Record | `docs/adr/` 决策记录 |
| DDD | Domain-Driven Design | GLOSSARY 即 ubiquitous language 文件化 |
| LLM | Large Language Model | 执行双 skill 调度的 agent |

## 为什么重要（对本知识库读者）

- **与 LLM Wiki 同构：** [Karpathy LLM Wiki](../references/llm-wiki-karpathy.md) 把知识编译进 `wiki/`；grill-with-docs 把 **会话中的决策** 编译进 **`GLOSSARY.md` + ADR** — 适合 **应用代码仓**，也可借鉴到大型 monorepo 工具脚本命名。
- **本库维护：** Robotics_Notebooks 已有 `schema/` 与 ingest 规范；若在 **fork 的应用层** 用 agent 改代码，宜用本技能统一 **术语**（sim2real、WBC 等）再改实现。

## 核心机制

| 调用 | 作用 |
|------|------|
| `grilling` | 设计树 frontier 追问 |
| `domain-modeling` | 挑战术语、lazy 创建 GLOSSARY/ADR、多 context 时 `GLOSSARY-MAP.md` |

## 常见误区或局限

- **误区：替代 wiki ingest。** 应用仓 ADR **不能** 代替本站 `sources/` + `wiki/` 分层。
- **前提：** 先 [setup-matt-pocock-skills](mattpocock-setup-skills-skill.md) 配置 `docs/adr/` 路径（若尚未 scaffold）。

## 关联页面

- [mattpocock/skills 总览](mattpocock-skills.md)
- [grill-me](mattpocock-grill-me-skill.md)
- [setup-matt-pocock-skills](mattpocock-setup-skills-skill.md)
- [tdd](mattpocock-tdd-skill.md) — 测试命名应读 GLOSSARY

## 参考来源

- [mattpocock/skills 归档](../../sources/repos/mattpocock-skills.md)
- [domain-modeling SKILL.md](https://github.com/mattpocock/skills/tree/main/skills/engineering/domain-modeling)

## 推荐继续阅读

- [course-video-manager CONTEXT.md 示例](https://github.com/mattpocock/course-video-manager/blob/076a5a7a182db0fe1e62971dd7a68bcadf010f1c/CONTEXT.md)
