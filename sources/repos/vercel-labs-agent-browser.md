# agent-browser（vercel-labs/agent-browser）

> 来源归档

- **标题：** agent-browser — Browser automation CLI for AI agents
- **类型：** repo
- **链接：** https://github.com/vercel-labs/agent-browser
- **分发：** https://skills.sh/vercel-labs/agent-browser/agent-browser（npm 包 `agent-browser`）
- **入库日期：** 2026-09-30
- **代码：** **已开源**（Apache-2.0；Rust CLI + Node 包装、Chrome for Testing 引导、skill 内容与 `agent-browser skills get` 动态文档均在 GitHub）
- **一句话说明：** 面向 coding agent 的 **原生 Rust 浏览器自动化 CLI**：CDP + 无障碍树快照 + `@eN` 元素引用；技能 stub 指向 `skills get core` 等与安装版本同步的工作流，并含 Electron/Slack/dogfood 等专项 skill。
- **为什么值得保留：** 与本站 ingest **步骤 2.5 项目页核查**、静态站 `docs/detail.html` 截图验证、以及 [BrowserSkill](browserskill.md)（借真实 Chrome profile）形成 **无头 CDP CLI vs 扩展借 tab** 对照；Cloud Agent 文档要求浏览器验证时可优先此 CLI 而非内置 web 工具。
- **沉淀到 wiki：** 是 → [`wiki/entities/vercel-agent-browser-skill.md`](../wiki/entities/vercel-agent-browser-skill.md)

## README 要点（归纳）

- **安装：** `npm i -g agent-browser && agent-browser install`（首次拉 Chrome for Testing）；支持 Homebrew、Cargo、pnpm 源码构建。
- **技能模型：** 根 `skills/agent-browser/SKILL.md` 为 **发现 stub**（`hidden: true`）；真实步骤经 `agent-browser skills get core|electron|slack|dogfood|...` 从 CLI 拉取，避免 SKILL 与版本漂移。
- **能力：** 会话、认证 vault、状态持久化、录像；可选 Vercel Sandbox / Bedrock AgentCore 云浏览器 skill。
- **观测：** 独立 dashboard（默认 4848），会话流量可经代理 URL 暴露。

## 关联资料

- skills.sh 页：[`sources/sites/skills-sh-agent-browser.md`](../sites/skills-sh-agent-browser.md)
- CLI 安装器：[`vercel-labs-skills.md`](vercel-labs-skills.md)
- 腾讯 BrowserSkill 对照：[`browserskill.md`](browserskill.md)
