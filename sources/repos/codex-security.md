# Codex Security（openai/codex-security）

> 来源归档（repo）

- **标题：** Codex Security — CLI & TypeScript SDK
- **URL：** <https://github.com/openai/codex-security>
- **npm：** <https://www.npmjs.com/package/@openai/codex-security>（包名 `@openai/codex-security`）
- **Homepage / 文档：** <https://developers.openai.com/codex/security>
- **类型：** repo / security-cli / typescript-sdk / application-security
- **License：** Apache-2.0
- **维护者 / 机构：** OpenAI（`openai`）
- **语言：** TypeScript（主）+ 捆绑 plugin（含 Python 脚本）
- **入库日期：** 2026-08-08
- **核查日规模：** 约 9.3k★ / 639 forks；发布 **0.1.8**（2026-08-07）
- **一句话说明：** OpenAI 开源的应用安全扫描 CLI 与 TypeScript SDK：用 Codex agent 发现、校验并辅助修复代码漏洞；支持 SARIF/CSV/JSON 导出、deep scan、容器化 bulk-scan 与 CI 严重度门禁。

## 开源核查（步骤 2.5）

| 维度 | 状态 |
|------|------|
| **开放程度** | **已开源（Apache-2.0）** — GitHub 完整源码 + npm 发布包 + 官方文档 |
| **训练权重** | 不适用（调用外部推理 API / ChatGPT 登录；非自研权重仓） |
| **可运行入口** | `npx @openai/codex-security scan|bulk-scan|export|validate|patch`；SDK `CodexSecurity.run()`；`docker compose` 批量扫描 |
| **访问门槛** | 需 Codex Security 访问权限；部分网络安全请求需 [Trusted Access for Cyber](https://chatgpt.com/cyber)；CI 用 `OPENAI_API_KEY` / `CODEX_API_KEY` |
| **文档** | [CLI](https://developers.openai.com/codex/security/cli)、[SDK](https://developers.openai.com/codex/security/sdk)、[SECURITY.md](https://github.com/openai/codex-security/blob/main/SECURITY.md) |

## 仓库结构（main，入库日）

| 路径 | 作用 |
|------|------|
| `README.md` | 根说明与 quick start |
| `sdk/typescript/` | ESM 包源码、`codex-security` 可执行文件、测试 |
| `sdk/typescript/_bundled_plugin/` | 捆绑 Codex Security plugin（schemas / MCP / deep-scan Python / 示例产物） |
| `Dockerfile` / `compose.yaml` / `compose.apparmor.yaml` | 官方容器化 bulk-scan |
| `docker/` | entrypoint、seccomp、可选 AppArmor |
| `.github/workflows/` | node-ci / container-ci / 发布流水线 |
| `SECURITY.md` / `CONTRIBUTING.md` | 安全披露与贡献指南 |

## 运行时要点（摘自官方 README）

- **运行时：** Node.js 22.13+（22.x）/ 24.x / 26.x；扫描与导出另需 Python 3.10+（3.10 需 `tomli`）。
- **认证：** 本地 `login`（ChatGPT / device-auth）；CI 环境变量 API key（不落盘到 credential home）；可选 OpenRouter / Fireworks / Amazon Bedrock provider。
- **扫描模式：** `standard` / `deep`（workers、subagents、stop-after-no-new、max-discovery-runs）；目标可为整仓、`--path`、`--diff`、`--working-tree`。
- **产物：** 结果目录须在 Git worktree **外**；`export` → SARIF / CSV / JSON；`scans compare` 按根因匹配 new / persisting / reopened / resolved。
- **CI 退出码：** 0 报告通过；1 严重度策略失败；2 输入无效 / 覆盖不全 / 运行时错误。
- **本地安全模型：** 以本机 OS 权限运行；`approvalPolicy: "never"`；仅扫描有权评估的仓库；结果可能含源码摘录与复现步骤。

## 对 wiki 的映射

- 主升格：[`wiki/entities/codex-security.md`](../../wiki/entities/codex-security.md)
- 文档站点：[`sources/sites/openai-codex-security-docs.md`](../sites/openai-codex-security-docs.md)
- 概念交叉：[`wiki/concepts/software-security-basics.md`](../../wiki/concepts/software-security-basics.md)、[`wiki/concepts/container-orchestration-cicd.md`](../../wiki/concepts/container-orchestration-cicd.md)
- 知识链：[`wiki/overview/hub-systems-engineering.md`](../../wiki/overview/hub-systems-engineering.md)
