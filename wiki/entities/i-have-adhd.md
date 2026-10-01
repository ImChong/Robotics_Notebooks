---
type: entity
tags:
  - llm-agents
  - coding-agents
  - skills
  - ux
  - agent-infrastructure
status: complete
updated: 2026-10-01
related:
  - ./caveman.md
  - ./ponytail.md
  - ./superpowers-obra.md
  - ./agent-skills-addyosmani.md
  - ../references/llm-wiki-karpathy.md
  - ../../schema/ingest-workflow.md
sources:
  - ../../sources/repos/i-have-adhd.md
summary: "i-have-adhd（ayghri/i-have-adhd）是编码代理输出结构技能：10 条规则让回复结论先行、步骤编号、抑制 tangent 并给出单一 next step；与 Caveman（短措辞）、Ponytail（少代码）正交，2026-10 Trendshift 约 +26.8k stars/月。"
---

# i-have-adhd

**i-have-adhd**（[ayghri/i-have-adhd](https://github.com/ayghri/i-have-adhd)）是安装到 Claude Code、Cursor 等 harness 的 **Agent Skill**：用固定 **10 条规则**  reshape 代理输出，使 **下一步行动、编号步骤与单一 follow-up** 出现在最前，减少「Great question!…Hope this helps!」类 filler。

## 一句话定义

用 **可版本化的 ADHD 友好输出契约** 管住编码 Agent 的 **信息顺序与密度**，让答案 **先可执行、后可展开**（不要求用户有 ADHD 诊断）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LLM | Large Language Model | 技能约束的对象模型输出 |
| SKILL.md | Agent Skill Manifest | `skills/i-have-adhd/SKILL.md` 完整规则文本 |
| UX | User Experience | 本技能优化的是人机阅读与执行体验 |

## 为什么重要（对本知识库读者）

- **与 Caveman / Ponytail 三角：**
  - [Caveman](caveman.md) — **更短措辞**（token）
  - [Ponytail](ponytail.md) — **更少代码**（LOC）
  - **i-have-adhd** — **更清晰的行动结构**（顺序与步骤）
  - [Superpowers（obra）](superpowers-obra.md) — **更对的交付流程**
- **维护本 wiki 的长会话：** ingest + `make ci-preflight` 常是多步；该技能降低 **在解释中丢失关键命令** 的概率，与 [Karpathy LLM Wiki](../references/llm-wiki-karpathy.md)「结论写进文件」可并用（文件写全、聊天说短）。

## 核心结构

| 组件 | 作用 |
|------|------|
| **SKILL.md** | 10 条规则：Lead with action、Number steps、One next step、Suppress tangents、Restate state、Time estimates、Visible wins 等 |
| **INSTALL.md** | 各 harness 安装说明 |
| **Before/After 表** | README 展示 auth 修复类任务的格式对比 |

## 局限

- 不替代 **测试、review、schema**；只改 **对话输出形状**。
- 极短输出可能与 **需要教学式推导** 的读者偏好冲突，可按任务关闭 skill。

## 关联页面

- [Caveman](caveman.md) — 输出 token 压缩
- [Ponytail](ponytail.md) — 必要性阶梯减代码
- [Agent Skills（Addy Osmani）](agent-skills-addyosmani.md) — 全 SDLC 技能包对照

## 参考来源

- [i-have-adhd 仓库归档](../../sources/repos/i-have-adhd.md)
- [ayghri/i-have-adhd（GitHub）](https://github.com/ayghri/i-have-adhd)

## 推荐继续阅读

- 上游 [SKILL.md](https://github.com/ayghri/i-have-adhd/blob/main/skills/i-have-adhd/SKILL.md)
- [Kacper Rutkiewicz 视频解读（README 链接）](https://youtu.be/NEl8kPWZP_Y)
