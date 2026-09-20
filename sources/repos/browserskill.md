# BrowserSkill（Tencent/BrowserSkill）

- **标题:** BrowserSkill
- **链接:** https://github.com/Tencent/BrowserSkill
- **类型:** repo / browser-automation / agent-skill
- **机构:** 腾讯（Tencent）
- **许可:** MIT
- **CLI:** `bsk`（`curl …/install.sh | sh` 安装至 `~/.local/bin`）
- **浏览器扩展:** [Chrome Web Store](https://chromewebstore.google.com/detail/hhcmgoofomhgciiibhipgmgkgnoenaoi) / [Edge Add-ons](https://microsoftedge.microsoft.com/addons/detail/browserskill/emacgiaaaiojkkpkddmmdfhmokgmnikg)
- **DeepSeek Harness 插件:** `@wxg-prc-cpg/browser-skill-dsh-plugin`（npm）
- **最后核查:** 2026-09-20
- **入库日期:** 2026-09-20

## 开源状态（步骤 2.5）

- **已开源：** GitHub 完整 monorepo（MIT）；`bsk` CLI + daemon + 扩展源码可本地构建；Chrome/Edge 商店分发扩展；dsh 插件已发 npm。无闭源模型权重依赖。

## 核心内容摘要

1. **复用真实登录态：** 代理通过 `bsk` 借用在用户已登录浏览器 profile 上的 tab，在独立 **Agent Window** 自动化，不抢占用户日常窗口。
2. **Harness 无关：** 任意能调 shell 的 agent（Cursor、Claude Code、Codex、OpenClaw、Hermes、DeepSeek Harness 等）经 `bsk` CLI 接入；内置 `browser-skill` SKILL.md，`bsk install-skill` 一键写入各 harness 技能目录。
3. **Human-in-loop：** captcha、二次确认、短信/人脸等步骤可 `request-help` 交还用户；扩展 popup 独立开关「借 tab 前确认 / 允许请求人工帮助」。
4. **架构：** Agent → `bsk` CLI → 本地 daemon（127.0.0.1 WebSocket）→ 浏览器扩展 → Agent Window；远程场景可经认证服务配对本地浏览器。
5. **工程栈：** Cargo + pnpm workspace（`crates/bsk-cli`、`apps/extension`、`packages/dsh-plugin-browserskill` 等）；含 `evals/browser` 确定性评测页。

## 对 wiki 的映射

- **wiki/entities/browserskill.md** — 框架实体（与 [deepseek-harness](../../wiki/entities/deepseek-harness.md)、[agent-reach](../../wiki/entities/agent-reach.md) 对照）
- **wiki/concepts/ai-auto-research.md** — 文献/项目页调研时的浏览器自动化层
- **wiki/references/llm-wiki-karpathy.md** — ingest 维护者借助 agent 读网页的工具体系
