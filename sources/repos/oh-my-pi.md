# oh-my-pi（can1357/oh-my-pi）官方仓库研读归档

> 来源归档（repo / 开源 Coding Agent）

- **标题：** oh-my-pi（OMP）— A coding agent with the IDE wired in
- **作者与维护：** Can Bölük / Stencil Labs；README 标注源自 Mario Zechner 的 [Pi](https://github.com/badlogic/pi-mono)
- **官方源码：** <https://github.com/can1357/oh-my-pi>
- **官方主页：** <https://omp.sh/>
- **官方 README：** <https://github.com/can1357/oh-my-pi/blob/main/README.md>
- **中文 README：** <https://github.com/can1357/oh-my-pi/blob/main/README.zh-CN.md>
- **源码许可证：** MIT（具体第三方组件另见各自许可证与 THIRD-PARTY-NOTICES）
- **类型：** repo / coding-agent / terminal-agent / LSP / DAP / MCP / ACP / Rust / TypeScript / Bun
- **入库日期：** 2026-11-21
- **核对范围：** 通读主仓库 README 的安装、21 项能力介绍、模型/检索适配、交互/RPC/ACP/SDK 入口、包结构、限制及许可；核查 `packages/coding-agent/package.json`、`src/cli.ts:runCli`、`src/cli-commands.ts`、`src/commands/launch.ts`、`src/main.ts:runRootCommand` 及 `src/sdk.ts` 的实际边界。未审计整个仓库，也没有执行 CLI 或 benchmarks。
- **通读：** 已完成
- **开源状态：** 已开源；GitHub 发布源代码，npm 包 `@oh-my-pi/pi-coding-agent` 的 `bin.omp` 指向 `src/cli.ts`（核查时 package.json 为 19.1.0；版本随时可能变化）。
- **一句话说明：** 基于 Pi 扩展的终端 Coding Agent，聚合 IDE 语言服务、调试、文件/代码操作、原生检索、子代理、会话记忆及多模型后端，不是机器人运动控制框架。
- **沉淀到 wiki：** [oh-my-pi 独立详情节点](../../wiki/entities/oh-my-pi.md)

## 官方 README 核心结构（2026-11-21 核对）

1. **定位与交互：** 带终端 TUI 的 Coding Agent；交互、一次性输出、RPC、ACP 与 SDK 是不同入口；项目以“IDE wired in”描述编辑器级能力。
2. **代码工具链：** 文件读取与编辑、bash/grep/glob、代码哈希锚定编辑；LSP 提供代码上下文和诊断，DAP 将调试会话和断点操作接入 agent 工具面。
3. **任务/会话：** `task` 子代理可在隔离工作区工作并返回结构化输出；可挂顾问模型、代码评审、长期记忆、协作会话及针对输出的规则反馈。
4. **外部与协议：** 支持多模型 Provider、网页检索/阅读、浏览器与桌面交互；支持 MCP 以及面向编辑器的 ACP。
5. **实现边界：** monorepo 包含 TypeScript/Bun CLI 和 Rust native crates；README 的性能数字及基准属于上游自述，本文未复测。
6. **安装入口：** macOS/Linux 官方 `curl -fsSL https://omp.sh/install | sh`、Bun `bun install -g @oh-my-pi/pi-coding-agent`、Homebrew `brew install can1357/tap/omp`、Nix；Windows PowerShell 官方另有安装脚本。执行前应确认来源与安装脚本权限。

## 可复核源码调用链（交互式 `omp`）

1. `packages/coding-agent/package.json`：npm `bin.omp` 指向 `src/cli.ts`；`src/cli.ts:runCli` 完成环境初始化、worker 分流，然后加载 `src/cli-commands.ts` 的 commands 表。
2. `src/cli-commands.ts`：常规参数映射到 `launch` 子命令；`src/commands/launch.ts:Index.run` 解析参数，再动态导入 `../main` 的 `runRootCommand`。
3. `src/main.ts:runRootCommand`：解析配置、身份凭证、模型与扩展；调用 `createAgentSession` 会话工厂（实现位于 `src/sdk.ts`）初始化会话；随后按运行模式分流。
4. **交互路径：** `runInteractiveMode(session, ...)` 启动 TUI；**一次性路径：** `./modes/print-mode` 的 `runPrintMode`；**RPC 路径：** `./modes/rpc/rpc-mode` 的 `runRpcMode`；ACP 路径则调用 `./modes/acp/acp-mode`。
5. **作用域说明：** 可确认上述入口和静态分流，但具体某个工具的执行是否调用 LSP/DAP、浏览器或子代理取决于用户任务与启用设置，不能画成每次调用都执行的强制流水线。

源码锚点：[package.json](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/package.json) · [cli.ts](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/src/cli.ts) · [cli-commands.ts](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/src/cli-commands.ts) · [launch.ts](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/src/commands/launch.ts) · [main.ts](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/src/main.ts) · [sdk.ts](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/src/sdk.ts)

## 与机器人研发知识库的关系

在机器人软件仓库，OMP 可作为**开发辅助工具**：审查 ROS 2/仿真代码、跨模块检索、重构与自动测试。但终端工具执行与 IDE 诊断 **不等于** 实测实时控制时延、仿真物理正确性或真机安全；物理系统仍需独立集成测试和人工审批。
