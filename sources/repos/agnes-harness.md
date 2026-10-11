# Agnes Harness（AgnesAI-Labs/agnes-harness）

- **标题：** Agnes Harness（AGH）
- **类型：** repo
- **来源：** AgnesAI-Labs
- **代码：** <https://github.com/AgnesAI-Labs/agnes-harness>
- **文档：** 仓库 README 与 `docs/`（含架构、安装、安全、MHS 设备指南）
- **许可：** 项目自有代码 Apache-2.0；第三方组件及部分改编文件按 `NOTICE` 与各自许可执行
- **入库日期：** 2026-10-11
- **复核日期：** 2026-10-11
- **一句话说明：** 面向 Forward Deployed Engineering（FDE）的 TypeScript agent harness，整合 Web、CLI 与 SDK 会话、工具/Skills/MCP 插件、审批和执行记录，并可通过自有 MHS 协议将设备接入任务流。
- **沉淀到 wiki：** 是 → [`wiki/entities/agnes-harness.md`](../../wiki/entities/agnes-harness.md)

## 开源状态核查（2026-10-11）

| 项 | 值 |
|----|-----|
| **开放程度** | **已开源（源码构建）** — 公开 TypeScript monorepo、examples、文档和 Apache-2.0 LICENSE |
| 发布阶段 | Developer preview / pre-alpha；尚无公开包发行、安装器或升级承诺 |
| 仓库结构 | `packages/`、`examples/`、`docs/`、`tools/`、`patches/`、`third-party/` |
| 构建环境 | 官方 README 要求 Node.js 24.10+、pnpm 10.34.5 与平台原生构建工具 |
| 运行入口 | README 给出 `pnpm install --frozen-lockfile`、`pnpm --filter @agnes/cli build:local`、`node packages/cli/dist/local/agnes.mjs serve` |
| 设备接入 | AgnesHub 是可选插件；AGH 的 MHS 设备协议通过 WebSocket 注册设备状态、工具与数据源 |
| 平台验证 | README 当前记录 macOS / Node 24 检查；Linux 与 Windows 仍需独立验收 |
| 外部贡献 | README 说明代码和文档 PR 目前限受邀内部开发者；Issues 可用于问题与用例反馈 |

核查依据是仓库当前 README、LICENSE、NOTICE 与运行/安全文档；未发现独立项目站。项目自有代码公开不等于已达到生产可用性，也不等于第三方目录内全部文件都采用 Apache-2.0。

## 核心架构（README / 架构指南）

- **共享入口：** Web workbench、CLI、终端 UI 与 SDK 使用同一后端会话，任务可跨 Web 与终端续接。
- **Agent Loop：** 管理模型/工具执行、任务状态、事件记录、中断处理与恢复；当前主分支使用内置 Core loop，Jev 结构化决策仍在集成中。
- **扩展面：** backend plugins、Web panels、Skills、hooks 和 MCP；Cordis 与包治理用于组织插件生命周期，示例覆盖工具、面板及 full-stack plugin。
- **可审计执行：** 命令默认先请求用户批准；Trace 记录模型调用、工具调用与审批。插件包安装/启用会显示来源、版本、完整性摘要、能力 hash 与许可证。
- **物理设备：** 可选 AgnesHub 插件让 agent 调用设备声明的工具；MOS（Model Observation Standard）描述相机图像、地图、地点等观测数据。

## 重要边界

- 普通 backend plugin 在进程内以可信代码运行；命令审批与 sandbox 只覆盖支持的执行路径，**不会隔离任意插件代码**。
- Agnes 的 **MHS（Model Hardware Standard）是本项目自己的设备协议**。它与 Anthropic / HHMI Janelia 的同名 Model Hardware Standard 独立定义，不能因缩写相同而当成兼容实现。
- MHS conformance suite 不代表设备安全认证。AGH 负责高层设备任务编排；实时控制、互锁与急停仍由设备控制器承担。取消 agent 任务也不证明设备已经安全停止。
- 项目仍是 pre-alpha；接口和配置可能变动，当前无生产承诺、公开包或可依赖升级路径。

## 对 wiki 的映射

- [Agnes Harness 实体页](../../wiki/entities/agnes-harness.md)
- [Model Hardware Standard 概念页](../../wiki/concepts/model-hardware-standard.md) — 特别注意两个独立协议使用相同全称与缩写
- [LLM 机器人控制接口](../../wiki/concepts/llm-robotics-control-interfaces.md)

## 官方入口

- README：<https://github.com/AgnesAI-Labs/agnes-harness>
- 架构指南：<https://github.com/AgnesAI-Labs/agnes-harness/blob/main/docs/guide/architecture.md>
- 安全与信任：<https://github.com/AgnesAI-Labs/agnes-harness/blob/main/docs/guide/security.md>
- MHS 与设备：<https://github.com/AgnesAI-Labs/agnes-harness/blob/main/docs/guide/mhs.md>
- 源码安装：<https://github.com/AgnesAI-Labs/agnes-harness/blob/main/docs/guide/install.md>
