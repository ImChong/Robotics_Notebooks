# AgentsDock / AgentsServer 官方 README 复核

- **类型：** repo
- **日期：** 2026-11-07
- **版本：** 当日读取两个仓库 main 根 README；滚动分支，不代表同一发行快照。
- **核对范围：** 完整阅读 AgentsDock 与 AgentsServer 根 README，涵盖定位、安装、开发、目录、更新、主机要求、内存门禁及许可。此记录不代表全仓源码审计或实机验收。
- **通读：** 已完成
- **原文：** https://github.com/ZhengyiLuo/AgentsDock/blob/main/README.md
- **配套原文：** https://github.com/ZhengyiLuo/AgentsServer/blob/main/README.md
- **映射：** [AgentsDock](../../wiki/entities/agentsdock.md)

## 复核结论

客户端以 Electron/React/TypeScript 与 React Native/Expo 提供跨端界面，服务器持有工作区与已登录的 Agent CLI。客户端 README 给出 Node.js 22.14+、uv、tmux 的部署准备步骤，以及 npm install -g agentsdock；源码开发要求 Node.js 24、pnpm 11.9.0。桌面 Gemini API 连接取决于集成版本，不应限定为仅支持四种 CLI。

独立后端 README 仍给出 ./install.sh、默认端口 7850、URL/token 连接、多实例与保留历史更新。默认 2 GiB/定时任务 4 GiB 是可用内存启动门禁，不是总内存推荐或运行保证。

两个 README 对后端归属有直接冲突：客户端称 server/ 为 maintained runtime，独立后端称客户端 server/ 为 compatibility fixtures。保留这项不一致，按实际选定版本部署；不能推定任一 main 自动适配另一方所有功能。两份 README 均声明自有代码 Apache-2.0，第三方许可独立。

原有产品站、旧发布与源码时序记录保持历史范围；本次只为根文档完整阅读补充明确证据，不声称已重新验证所有历史功能或源码调用。
