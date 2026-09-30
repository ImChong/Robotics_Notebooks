---
type: entity
tags:
  - llm-agents
  - coding-agents
  - skills
  - test-driven-development
  - software-engineering
status: complete
updated: 2026-09-30
related:
  - ./mattpocock-skills.md
  - ./mattpocock-grill-with-docs-skill.md
  - ./superpowers-obra.md
  - ./agent-skills-addyosmani.md
  - ../concepts/agentic-coding-software-fundamentals.md
  - ../../schema/ingest-workflow.md
sources:
  - ../../sources/repos/mattpocock-skills.md
  - ../../sources/sites/skills-sh-mattpocock-selected-skills.md
summary: "tdd 技能把 RED-GREEN 垂直切片、seam 共识、反模式（实现耦合/同义反复/横向切分）写进 SKILL.md；与本站 make ci-preflight 文化一致，适用于 scripts/ 与 docs/ 工具链修改。"
---

# tdd（Matt Pocock Skill）

**tdd**（[skills.sh](https://skills.sh/mattpocock/skills/tdd)）是 [mattpocock/skills](https://github.com/mattpocock/skills) 的 **测试驱动开发** 规约：强调 **垂直切片**、在 **已确认 seam** 上写行为规格测试，并列出 agent 常犯的 **horizontal slicing** 与 **tautological** 断言反模式。

## 一句话定义

**先红后绿、一次一片**：每个循环只做一个 seam 的失败测试 + 最小实现；重构留给 code-review 技能而非混进红绿环。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TDD | Test-Driven Development | 红 → 绿 →（重构外置） |
| RED | （测试失败阶段） | 先写失败测试 |
| GREEN | （测试通过阶段） | 最小实现通过 |

## 为什么重要（对本知识库读者）

- **本站 CI：** `make ci-preflight` / `make test` 是 **仓库级 seam**；用本 skill 改 `scripts/*.py` 时，应先 **与用户确认测哪些公共行为**（export、lint、graph），避免 mock 内部实现导致 refactor 即红。
- **与 Superpowers / Addy：** [Superpowers](superpowers-obra.md) **强制** TDD 管线；[Addy Osmani](agent-skills-addyosmani.md) 有 `/test` 命令；mattpocock **tdd** 更细 **seam 与反模式** 文本。

## 核心规则（摘要）

| 主题 | 要求 |
|------|------|
| Seam | 写测试前 **书面列出 seam 并与用户确认** |
| 接口未定 | 调 `codebase-design` 技能对齐 module/interface 词汇 |
| 反模式 | 实现耦合、tautological 期望、先写全套测试再实现 |
| 词汇 | 读 `GLOSSARY.md`；与 [grill-with-docs](mattpocock-grill-with-docs-skill.md) 联动 |

## 常见误区或局限

- **误区：测 private 方法。** skill 明确测 **公共行为**。
- **局限：** 真机/仿真 E2E 成本高 — seam 应选在 **可重复** 的脚本/单元层。

## 关联页面

- [mattpocock/skills 总览](mattpocock-skills.md)
- [improve-codebase-architecture](mattpocock-improve-codebase-architecture-skill.md) — 找 deep module 后再 TDD
- [Ingest Workflow](../../schema/ingest-workflow.md)

## 参考来源

- [tdd SKILL.md](https://github.com/mattpocock/skills/tree/main/skills/engineering/tdd)
- [mattpocock/skills 归档](../../sources/repos/mattpocock-skills.md)

## 推荐继续阅读

- [skills.sh tdd](https://skills.sh/mattpocock/skills/tdd)
