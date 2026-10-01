---
type: entity
tags:
  - llm-agents
  - coding-agents
  - skills
  - agent-infrastructure
  - security
  - open-source
status: complete
updated: 2026-10-01
related:
  - ./superpowers-obra.md
  - ./agent-skills-addyosmani.md
  - ./open-code-review.md
  - ./hindsight.md
  - ./ponytail.md
  - ./mattpocock-skills.md
  - ../references/llm-wiki-karpathy.md
  - ../../schema/ingest-workflow.md
sources:
  - ../../sources/repos/ecc.md
  - ../../sources/sites/ecc-tools.md
summary: "ECC（affaan-m/ECC）是 MIT 开源的 Agent harness 工程操作系统：plan→test→implement→review→verify→remember→improve 闭环，打包 68 agents、293 skills、hooks、continuous learning 与 AgentShield；跨 Claude Code/Codex/Cursor 等，2026-10 约 270k stars、Trendshift +26.8k/月。"
---

# ECC（Everything Claude Code）

**ECC**（[affaan-m/ECC](https://github.com/affaan-m/ECC)，[ecc.tools](https://ecc.tools)）是把 **编码 Agent 的工程习惯** 打包成 **可安装系统** 的开源项目（MIT）：默认叙事是 **优化 context window，把其余一切持久化** — 计划、测试、实现、自审、验证、记忆与持续改进 — 而不是每轮 prompt 里重写流程。

## 一句话定义

一次安装给 Agent **协调好的工程工具箱 + 运行时 hooks + 安全扫描**，把 **skills-first SDLC** 从个人 prompt 变成 **可版本化的 harness 扩展**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ECC | Everything Claude Code（社区扩展义：跨 harness Agent 工程系统） | 本仓库品牌名 |
| SDLC | Software Development Lifecycle | plan/test/review 等阶段覆盖 |
| TDD | Test-Driven Development | skills 库中的核心实践之一 |
| MCP | Model Context Protocol | 与 AgentShield 等安全/工具扩展相关 |

## 核心信息

| 字段 | 内容 |
|------|------|
| 许可 | **MIT**（OSS）；ECC Pro GitHub App 为私有仓托管 |
| Stars（2026-10-01） | ~270k（Trendshift 约 +26.8k/月） |
| 推荐安装 | `npx ecc-universal@2.2.2 setup`（以 README 当前版为准） |
| 安全 | **AgentShield** — 扫描 prompt、hooks、MCP、secrets |

## 为什么重要（对本知识库读者）

- **规模对照：** 与 [Superpowers（obra）](superpowers-obra.md)、[Addy Osmani Agent Skills](agent-skills-addyosmani.md) 同属 **Agent 行为文件化** 路线，但 ECC 以 **agent/skill 数量 + 跨 7+ harness 适配** 成为生态 **默认提及** 之一；选型时需读 upstream **platform support 矩阵**，勿假设 Cursor 与 Claude Code 功能 parity。
- **与本仓库 CI 同向：** `make ci-preflight` 已是确定性门禁；ECC 的 **test/review/verify** 技能可嵌入 **人类+Agent 共维护 wiki** 的会话（仍不能替代 schema lint）。
- **记忆层：** 与 [Hindsight](hindsight.md) 等 **外部记忆服务** 可组合 — ECC 偏 **harness 内 instincts/continuous learning**，Hindsight 偏 **长期 retain/recall/reflect**。

## 核心结构

| 包 | 规模（README 表） | 作用 |
|----|-------------------|------|
| Agents | 68 | 规划、review、build repair、安全、架构等角色 |
| Skills | 293 | TDD、研究、文档、前端、ML、运维等 |
| Commands | 94 | 向 skills-first 过渡的快捷入口 |
| Hooks / memory | Runtime | 会话摘要、continuous learning、instincts |
| AgentShield | 安全 | 代理文件与配置扫描 |

```mermaid
flowchart LR
  P[plan] --> T[test]
  T --> I[implement]
  I --> R[review]
  R --> V[verify]
  V --> M[remember]
  M --> X[improve]
```

## 局限与风险

- **仅官方渠道安装** — README 警告第三方镜像风险。
- **Harness 能力不齐** — Cursor/OpenCode 等为 **capability-limited adapters**。
- **体量过大** — 293 skills 需要团队 **选择性启用**，否则 context 反噬。

## 关联页面

- [Superpowers（obra）](superpowers-obra.md) — 流程技能对照
- [Open Code Review](open-code-review.md) — 行级 review 工具链
- [Hindsight](hindsight.md) — 学习型记忆
- [Ponytail](ponytail.md) — 减 over-engineering 代码

## 参考来源

- [ECC 仓库归档](../../sources/repos/ecc.md)
- [ecc.tools 站点归档](../../sources/sites/ecc-tools.md)
- [affaan-m/ECC（GitHub）](https://github.com/affaan-m/ECC)

## 推荐继续阅读

- 上游 [Install ECC](https://github.com/affaan-m/ECC#install-ecc) 与 platform support 矩阵
- [Superpowers 实体页](superpowers-obra.md) — 更小但深的流程技能包对照
