---
type: entity
tags:
  - tsinghua
  - llm-agents
  - multi-agent
  - education
  - open-source
status: complete
updated: 2026-10-01
related:
  - ./openclaw.md
  - ./paperclip.md
  - ./hyperframes.md
  - ../references/llm-wiki-karpathy.md
  - ../../schema/ingest-workflow.md
sources:
  - ../../sources/repos/openmaic.md
  - ../../sources/sites/openmaic-live.md
summary: "OpenMAIC（THU-MAIC/OpenMAIC）是清华 MIT 开源的多智能体互动课堂：一键或 Agent Workbench 从材料生成幻灯片/测验/交互/PBL，AI 师生 TTS 与白板；LangGraph 编排，OpenMAIC Skill 可接 OpenClaw 与 IDE，2026-10 Trendshift 约 +18.1k/月。"
---

# OpenMAIC

**OpenMAIC**（[THU-MAIC/OpenMAIC](https://github.com/THU-MAIC/OpenMAIC)，Demo [open.maic.chat](https://open.maic.chat/)）是 **Open Multi-Agent Interactive Classroom**：用 **多 Agent 编排** 把主题或文档变成 **可播放的沉浸式课堂** — 幻灯片、测验、交互 HTML 仿真、PBL — 由 **AI 教师与 AI 同学** 讲解、讨论、白板与 TTS。

## 一句话定义

**一键（或 Workbench 对话）生成多 Agent 课堂体验**，并把 **课程构建本身** 做成可持久化、可Steer 的 Agent 任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MAIC | Multi-Agent Interactive Classroom | 项目全称核心 |
| PBL | Project-Based Learning | 项目式学习场景类型 |
| TTS | Text-to-Speech | AI 角色语音讲解 |
| ASR | Automatic Speech Recognition | 可选 FunASR 等本地语音识别 |

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 清华大学（Tsinghua）MAIC |
| 许可 | **MIT**（v0.3.0 起重许可自 AGPL 变更，见 CHANGELOG） |
| 栈 | Next.js 16、React 19、LangGraph、Postgres |
| Stars（2026-10-01） | ~39.7k（Trendshift 约 +18.1k/月） |

## 为什么重要（对本知识库读者）

- **机器人/wiki 教学：** 可把 [roadmap](../../roadmap/) 或 **单篇 wiki** 喂给 OpenMAIC 生成 **带测验与交互** 的 onboarding（仍须人工 fact-check 对齐 schema）。
- **多 Agent 工程样本：** LangGraph + **24+ course skills** + **安全加固迭代**（2026-09 多个 GHSA）是 **教育类 Agent 产品** 的公开参考。
- **OpenClaw 集成：** `skills/openmaic/SKILL.md` — 与 [OpenClaw](openclaw.md) 从 IM/IDE **远程生成课堂** 对齐。

## 核心结构

| 模式 | 说明 |
|------|------|
| **Classic one-click** | 描述主题 → 自动生成课程 |
| **Agent Workbench（v1.0+）** | Chat-first 规划/修订全课；server-backed session |
| **Hosted vs self-host** | access code 或 Vercel/Docker 自部署 |
| **导出** | `.pptx`、交互 `.html`、视频导出（见 CHANGELOG） |

## 局限

- 需 **LLM provider 密钥** 与（推荐）外部 **Postgres**；成本随模型与媒体生成上升。
- 生成内容 **非自动符合** 本库 canonical-facts；入库前需人审。
- 升级 v1.1+ 前必读 **Behavior Changes**（默认 chat runtime 等）。

## 关联页面

- [OpenClaw](openclaw.md) — OpenMAIC Skill 宿主之一
- [Paperclip](paperclip.md) — 多 Agent 工作编排（偏商业目标）
- [Hyperframes](hyperframes.md) — 课程视频导出可对照

## 参考来源

- [OpenMAIC 仓库归档](../../sources/repos/openmaic.md)
- [open.maic.chat 站点归档](../../sources/sites/openmaic-live.md)
- [THU-MAIC/OpenMAIC（GitHub）](https://github.com/THU-MAIC/OpenMAIC)

## 推荐继续阅读

- [JCST'26 论文](https://jcst.ict.ac.cn/en/article/doi/10.1007/s11390-025-6000-0)
- 上游 [User Guide（飞书）](https://lcn6dqn3m0yr.feishu.cn/wiki/CkQSwHFdzibQFvkGzwPcmUOfnXg)
