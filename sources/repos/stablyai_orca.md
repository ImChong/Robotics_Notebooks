# Orca 官方源码与运行文档

- **源码：** [stablyai/orca 固定提交 5ae7183](https://github.com/stablyai/orca/tree/5ae71836b832c3e259b35f0e24ffcb585ccc2030)，该提交 package.json 版本为 1.4.214，不代表各平台当前商店或发行版版本。
- **发布方与许可：** GitHub 组织及 package.json 标为 Stably AI；仓库 [LICENSE](https://github.com/stablyai/orca/blob/5ae71836b832c3e259b35f0e24ffcb585ccc2030/LICENSE) 为 MIT，版权页列 Lovecast Inc.。第三方模型、客户端服务及依赖不能沿用这一许可。
- **官网：** [Orca](https://orca.dev/) · [执行主机选型](https://orca.dev/docs/ways-to-run) · [官方 Bot 文档](https://orca.dev/docs/bots)。
- **核对日期：** 2026-11-26
- **核对范围：** 固定提交 README（全篇）、LICENSE、package.json、worktree CLI 和主机 RPC/启动协作入口；对照官方方式文档的本机、SSH、远程 Server 和按工作区环境。未跑安装、远程配对、真实模型调用或 PR 自动化。
- **通读：** 已完成

## 归档要点

- [worktree.ts](https://github.com/stablyai/orca/blob/5ae71836b832c3e259b35f0e24ffcb585ccc2030/src/cli/workspaces/worktree.ts) 校验工作区及启动选项；无启动 Agent 时发 `worktree.create`。
- [worktree-create-agent-launch.ts](https://github.com/stablyai/orca/blob/5ae71836b832c3e259b35f0e24ffcb585ccc2030/src/cli/workspaces/worktree-create-agent-launch.ts) 根据主机能力选择 `agent.launch` 或兼容的 `worktree.create` 启动字段；旧主机不支持的模型、推理配置或界面不能保证降级无损。
- [agent-launch.ts](https://github.com/stablyai/orca/blob/5ae71836b832c3e259b35f0e24ffcb585ccc2030/src/main/runtime/rpc/methods/agent-launch/agent-launch.ts) 将调用交给主机侧启动流程；[agent-launch-executor.ts](https://github.com/stablyai/orca/blob/5ae71836b832c3e259b35f0e24ffcb585ccc2030/src/main/agent-launch/agent-launch-executor.ts) 先准备工作区，再根据已协商模式建立聊天会话或终端。此处仅为静态调用关系。
- README 所列 Claude Code、Codex、Grok、Antigravity 等为可用 CLI 示例；原生聊天、Bot 集成、恢复能力不等同于整个 CLI 列表。执行端需有可用客户端／账号；自付模型用量，不承诺不限额或实现订阅共享。
- **映射：** [Orca Wiki 节点](../../wiki/entities/stablyai-orca.md)。
