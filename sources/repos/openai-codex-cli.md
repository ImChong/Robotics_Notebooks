# OpenAI Codex CLI（openai/codex）

> 来源归档（repo）

- **名称：** Codex CLI
- **类型：** repo / coding-agent / cli / rust / developer-tools / openai
- **URL：** <https://github.com/openai/codex>
- **官方文档：** <https://developers.openai.com/codex>
- **许可证：** Apache-2.0
- **入库日期：** 2026-10-03
- **一句话说明：** OpenAI 开源的本地编码代理命令行实现，包含 Rust 工作区、跨平台 CLI 安装包装层与调用 CLI 的 TypeScript SDK。

## 仓库结构与运行方式

| 路径 | 用途 |
|------|------|
| `codex-rs/` | Codex CLI 的 Rust workspace 与多个功能 crate |
| `codex-cli/` | npm CLI 启动包装层；依据操作系统与架构选择平台二进制 |
| `sdk/typescript/` | TypeScript SDK；启动 CLI 并通过 stdin/stdout 交换 JSONL 事件 |
| `docs/` | 安装、配置、执行策略与使用文档 |

仓库根 README 将 Codex CLI 定位为运行在用户计算机上的编码代理，并区分 IDE 集成和云端 Codex Web。README 提供安装脚本、npm、Homebrew 和 GitHub Release 等分发方式；具体平台支持以当前 release 为准。

## 开放边界（2026-10-03）

- **已开源：** CLI 源码、Rust 实现、TypeScript SDK 与相关文档；项目采用 Apache-2.0。
- **未包含：** 代码仓库不等于模型权重或推理服务。README 说明可通过 ChatGPT 订阅登录，也可配置 OpenAI API key；底层模型与在线服务不因 CLI 源码开放而一并开源。
- **使用边界：** Codex CLI 读写本地工作目录并运行开发命令；安全沙箱与审批策略见官方 [security 文档](https://developers.openai.com/codex/security) 和仓内 `docs/execpolicy.md`。
- **相邻产品：** 本仓库是 CLI 源码，和 [Codex Security](../../wiki/entities/codex-security.md)（专用 AppSec CLI / SDK）及 Codex Web 的任务界面不同。
