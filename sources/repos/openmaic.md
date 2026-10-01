# OpenMAIC（THU-MAIC/OpenMAIC）

> 来源归档

- **标题：** OpenMAIC — Open Multi-Agent Interactive Classroom
- **类型：** repo（Next.js 多 Agent 互动课堂平台 + Agent Workbench + Skill）
- **机构：** 清华大学 MAIC（THU-MAIC）
- **链接：** https://github.com/THU-MAIC/OpenMAIC
- **在线 Demo：** https://open.maic.chat/ — [`sources/sites/openmaic-live.md`](../sites/openmaic-live.md)
- **论文：** JCST'26 — <https://jcst.ict.ac.cn/en/article/doi/10.1007/s11390-025-6000-0>
- **许可：** MIT（v0.3.0 起由 AGPL-3.0 变更，见 CHANGELOG）
- **入库日期：** 2026-10-01
- **Trendshift（用户触发，2026-10）：** 约 **+18.1k stars/月**；GitHub API 2026-10-01 约 **39.7k** stars
- **一句话说明：** 清华开源 **多智能体互动课堂**：一键或 Agent Workbench 从主题/文档生成 **幻灯片、测验、交互 HTML、PBL**；AI 教师与同学 **TTS、白板、实时讨论**；v1.0+ 支持 **持久 session、材料上传、24+ 内置 skills**；OpenMAIC Skill 可接 OpenClaw / Codex / IDE workbench。
- **为什么值得保留：** 与机器人 wiki 的 **教学/ onboarding** 场景相邻；展示 **LangGraph 多 Agent 编排 + 课程 DSL + 安全加固迭代**（2026-09 多版 security release）的可复用范式。
- **沉淀到 wiki：** 是 → [`wiki/entities/openmaic.md`](../../wiki/entities/openmaic.md)

## README 要点（归纳，2026-10-01）

- **栈：** Next.js 16、React 19、TypeScript、LangGraph 1.1、Tailwind 4；Postgres 持久化；Vercel 一键部署模板。
- **v1.0.0（2026-08-27）：** Agent workbench、server-backed sessions、session materials、course tools。
- **v1.1.x：** 课堂 chat 改为 agent loop；Pi 默认 chat runtime（升级前读 Behavior Changes）。
- **集成：** OpenClaw `clawhub install openmaic`；Hosted mode access code 或 self-host。
- **本地 AI：** Lemonade、FunASR 等可选路径（README 专节）。

## 开源状态

- **已开源（MIT）** — 完整应用与 `skills/openmaic/`；需自备 LLM API 与外部 Postgres（部署文档 `.env.example`）。

## 对 wiki 的映射

| 目标 | 链接 |
|------|------|
| 实体页 | [`wiki/entities/openmaic.md`](../../wiki/entities/openmaic.md) |
| OpenClaw | [`wiki/entities/openclaw.md`](../../wiki/entities/openclaw.md) |
