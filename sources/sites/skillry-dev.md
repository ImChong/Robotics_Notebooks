# Skillry（skillry.dev）

- **标题：** Skillry — Agent Skills with taste
- **类型：** site / marketplace
- **URL：** <https://skillry.dev>
- **入库日期：** 2026-09-30
- **代码：** 截至入库日 **无公开 GitHub 仓库**；技能包经官方 CLI 在登录后流式下发（见 <https://skillry.dev/install/agent.md>）
- **开源状态：** **未开源**（商业 Skill 目录 + 订阅；单 Skill 安装包为私有归档，非 MIT 公开仓）

## 一句话摘要

约 **150 个精选 Agent Skill** 的 **交付物导向市场**：按 **Web / Slides / Image / Video** 四类产出组织，强调 **先看页面效果与安装量再安装**；通过 `skillry-cli` 浏览器 OAuth 连接账户，月付约 **$9.99**（站点称 founding price，随库增长可能调整）。

## 公开信息要点（截至入库日 2026-09-30）

- **定位：** 「Agent Skills with taste」— 与通用 skills.sh 排行榜不同，偏 **成品工作流**（landing、deck、OG 图、片头等），非纯编码 SDLC。
- **兼容代理：** 首页宣称 Claude Code、Codex、Cursor 等 **+6** 类 harness；安装路径 **因 Skill 而异**（每 Skill 有 `/skills/<slug>/install.md`，**不得**假设统一为 `~/.agents/skills`）。
- **连接方式：** `npx --yes skillry-cli@latest login` → 浏览器授权 → `status`；**不要求用户粘贴 API key**；凭证存 OS keyring。
- **访问模型：** 目录可浏览；**Free 与 Premium Skill 安装均需登录**；服务端校验 entitlement 后流式私有 ZIP。
- **定价（页显）：** 月订 **$9.99**，可随时取消；年付文案为 founding price。
- **发现入口：** 站点 Featured / Top-rated 分栏；代理可读 `https://skillry.dev/install/agent.md` 引导连接 CLI 后选 Skill。

## 为何值得保留

- **与开源「反 slop」技能互补：** [Taste Skill](../../wiki/entities/taste-skill.md)、[Impeccable](../../wiki/entities/impeccable.md) 治理 **生成约束**；Skillry 提供 **可复用的交付物模板市场**，适合「先要能看的 landing/deck，再谈设计系统」的场景。
- **安装安全模型：** 官方 agent.md 明示 **勿执行 Skill 包内嵌指令** 于安装流之外 — 可作为 wiki 讨论 **不可信 Skill 包** 的参照。

## 关联资料

- wiki：[`wiki/entities/skillry.md`](../../wiki/entities/skillry.md)
- 对照：[`wiki/comparisons/skillry-taste-skill-impeccable.md`](../../wiki/comparisons/skillry-taste-skill-impeccable.md)
