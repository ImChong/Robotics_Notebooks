# DeepSeek Harness（deepseek-ai/deepseek-harness）

- **标题：** DeepSeek Harness（`dsh`）
- **类型：** repo
- **来源：** DeepSeek AI（[deepseek-ai](https://github.com/deepseek-ai)）
- **链接：** <https://github.com/deepseek-ai/deepseek-harness>
- **npm：** `@deepseek-ai/dsh`（`npx @deepseek-ai/dsh web`）
- **Python SDK：** `deepseek-harness-sdk`（`from deepseek_harness import DeepSeekHarness`）
- **项目页：** 无独立 `*.github.io` / 产品站；文档在仓内 `docs/` + `website/`（VitePress）
- **入库日期：** 2026-08-13
- **一句话说明：** DeepSeek 官方 **插件化 agent harness**：一切皆插件，内核为 vendored [Cordis](https://github.com/cordiverse/cordis)；提供 Web UI、headless CLI、Python SDK 与 ACP。
- **沉淀到 wiki：** 是 → [`wiki/entities/deepseek-harness.md`](../../wiki/entities/deepseek-harness.md)

## 开源状态核查（2026-08-13）

| 项 | 值 |
|----|-----|
| **开放程度** | **已开源** — MIT；完整 TypeScript monorepo（`packages/`）、Python SDK（`python/`）、Landlock native addon、VitePress 文档与可运行示例 |
| Stars / Forks（API） | ~16,178 / ~1,055 |
| 默认分支 | `master` |
| 主要语言 | TypeScript（Node `^22.19 \|\| >=24`）；另有 Python SDK 与 `native/` |
| 版本（根 `package.json`） | **0.1.0-rc.5**（开发者预览；README 声明将出现破坏兼容变更） |
| 许可 | **MIT**（`LICENSE` + `THIRD_PARTY_NOTICES.md`） |
| 权重 / 模型 | **不自带** 模型权重；默认走 DeepSeek API（`DEEPSEEK_API_KEY`），目录另含 Anthropic / OpenAI 等 provider，可加 OpenAI-compatible 自定义端点 |
| 项目页 | **无**（GitHub `homepage` 为空）；Issues / PRs 关闭，反馈走 [Discussions](https://github.com/deepseek-ai/deepseek-harness/discussions) |
| 插件发现 | GitHub topic [`dsh-plugin`](https://github.com/topics/dsh-plugin) |

步骤 2.5：无独立项目页。源码以本仓为准，README / `docs/architecture.md` / `docs/user/guide/` 给出可运行入口 → **已开源**。

## 仓库概况（2026-08-13 API / README）

| 字段 | 值 |
|------|-----|
| 描述 | （API `description` 为空；README：open-source agent harness，everything is a plugin） |
| Topics | 空 |
| 创建 | 2026-08-13 |
| Issue / PR | **关闭**（`has_issues=false`，`has_pull_requests=false`）；`has_discussions=true` |
| 包管理 | pnpm 11.7 workspaces |

## README 摘要

> DeepSeek Harness (`dsh`) is an open-source agent harness developed by DeepSeek AI. It uses an architecture where **everything is a plugin**, and is powered by [Cordis](https://github.com/cordiverse/cordis).

**运行入口（README）：**

1. `npx @deepseek-ai/dsh web` → Web UI，默认 `http://127.0.0.1:3080`
2. 源码：`pnpm install && pnpm run build && pnpm dsh web`
3. Headless：`pnpm dsh --profile headless "task"`（需 `DEEPSEEK_API_KEY`）
4. Python：`pip install deepseek-harness-sdk`，见 `docs/user/guide/python-sdk.md`

## 仓库结构要点（2026-08-13 `AGENTS.md` / tree）

| 路径 | 角色 |
|------|------|
| `vendor/` | 钉死的 Cordis 源码副本（rescoped `@deepseek-ai/cordis`） |
| `packages/core/` | 产品 API 脊柱：session、system-prompt、tools、agent、agent-loop |
| `packages/llm/` | LLM 能力缝 + DeepSeek / 目录 provider |
| `packages/bundle/` | `dsh-base` / `dsh-web-app` / `dsh-headless` 等 profile 层 |
| `packages/skill/` | skill provider + catalog/loader |
| `packages/subagent/` | 子代理委托 |
| `packages/e2b/` | E2B 沙箱 + FS/subprocess 适配（POC） |
| `packages/acp/` | Agent Client Protocol 自动化服务 |
| `packages/sdk/` | JSON-RPC 协议、服务端与 TypeScript 客户端 |
| `packages/hooks/` | Claude Code / Codex hook 桥 |
| `apps/cli/` | `dsh` CLI |
| `python/` | 已发布 Python SDK + 捆绑运行时（安装后不需本机 Node） |
| `native/` | `@deepseek-ai/node-addon-landlock-run` |
| `docs/` | 架构、turn/step 生命周期、cookbook、用户指南 |
| `website/` | 仓内 VitePress 文档站 |
| `examples/` | 可运行 `cordis.yml` 叶子（含 `jsonrpc-agent`） |

## 与机器人研究/工程的关联点

- **Coding agent 后端：** Web / headless / Python SDK 可编排仓库级任务（读改文件、bash、子代理），是 [真机策略 autoresearch](../../wiki/queries/real-robot-policy-autoresearch-harness.md) 里 Codex / Claude Code / Kimi Code 之外的 **官方 DeepSeek 宿主**。
- **不是运动栈：** 不发布策略、仿真器或 Robot Gateway；物理执行仍走本库控制 / VLA / 导航页。
- **同名勿混：** 本仓是 **LLM agent 运行时**。具身侧的 [Harness VLA](../../wiki/entities/paper-harness-vla.md)（RPent）与 [RoboHarness](../../wiki/entities/paper-robo-harness.md) 是 **冻结 VLA + planner 编排**，问题不同。
- **插件缝：** Cordis 的 Service Definition / Provider / Consumer 与「模型可见 ⟺ 已入 session log」不变量，可对照本库维护 agent 时的工具/日志契约。

## 对 wiki 的映射

- 升格页面：[wiki/entities/deepseek-harness.md](../../wiki/entities/deepseek-harness.md)
- 交叉引用：[wiki/entities/hermes-agent.md](../../wiki/entities/hermes-agent.md)、[wiki/entities/openclaw.md](../../wiki/entities/openclaw.md)、[wiki/entities/kimi-k3.md](../../wiki/entities/kimi-k3.md)、[wiki/queries/real-robot-policy-autoresearch-harness.md](../../wiki/queries/real-robot-policy-autoresearch-harness.md)、[wiki/entities/paper-harness-vla.md](../../wiki/entities/paper-harness-vla.md)

## 参考链接

- 源码仓库：<https://github.com/deepseek-ai/deepseek-harness>
- 架构文档：<https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md>
- Web UI 指南：<https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/index.md>
- Python SDK 指南：<https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/python-sdk.md>
- Cordis：<https://github.com/cordiverse/cordis>
- Cordis 设计论文：<https://github.com/cordiverse/paper>
- Discussions：<https://github.com/deepseek-ai/deepseek-harness/discussions>
