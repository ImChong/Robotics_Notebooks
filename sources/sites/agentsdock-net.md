# AgentsDock 产品站 — agentsdock.net

- **类型：** 产品网站 / 用户文档（静态 HTML）
- **主链接：** <https://agentsdock.net/>
- **代码（客户端）：** <https://github.com/ZhengyiLuo/AgentsDock>
- **代码（后端）：** <https://github.com/ZhengyiLuo/AgentsServer>
- **收录日期：** 2026-10-01
- **开源状态（步骤 2.5，2026-10-01 核查）：** **已开源** — 项目页 Hero 区链 GitHub；客户端与后端均为 Apache-2.0，桌面/移动安装包与服务器安装脚本可公开下载；模型推理仍走用户自选的 Claude/Codex/Cursor 等 **云端提供商**，非本地模型。

## 一句话

面向 **agentic AI 研究** 的跨端 IDE：**AgentsDock** 为 Electron / React Native 客户端，**AgentsServer** 为自托管执行后端；在同一工作区里调度 **Claude Code、Codex、Cursor、OpenCode（beta）** 等已安装在服务器上的编码代理，并支持文件预览、tmux 终端、定时任务与多设备接入。

## 为什么值得保留

- **客户端–服务器边界清晰**：代理 CLI 与项目目录留在 **服务器**；手机/平板只做连接与审阅，适合长时跑实验、仿真或训练脚本时远程盯进度。
- **与人形 / 机器人研究作者的交叉**：维护者为 [Zhengyi Luo](../../wiki/entities/zhengyi-luo.md)（NVIDIA GEAR）；与本站 **编码代理基础设施**（[Hermes Agent](../../wiki/entities/hermes-agent.md)、[Agent Reach](../../wiki/entities/agent-reach.md)）形成「IDE 壳 + 自托管后端」对照。
- **Git 工作区能力在演进**：Stable **1.0.3** 起客户端 **Workspace Changes** 可在 UI 内 stage/commit/解冲突（需 AgentsServer 1.0.3）；**尚不**创建分支合并、push 或管理 GitHub Pull Request（以 [RELEASE_1.0.3](https://github.com/ZhengyiLuo/AgentsDock/blob/main/docs/RELEASE_1.0.3.md) 为准）。

## 站点要点（2026-10-01 抓取归纳）

| 区块 | 内容 |
|------|------|
| **定位** | “An IDE designed for agentic AI research” |
| **下载** | macOS / Linux x86_64·ARM64 / Windows；iOS App Store；Android APK（独立 Releases 仓） |
| **文档** | [setup.html](https://agentsdock.net/setup.html)、[features.html](https://agentsdock.net/features.html)、[update.html](https://agentsdock.net/update.html) |
| **社区** | [Discord](https://discord.gg/ZGDrhEWqPt) |
| **功能摘要** | 导入本机 Claude/Codex 会话、fork 聊天、Side chat、Agent-to-agent 消息、定时任务、远程文件浏览/编辑、每聊天 tmux 终端、多服务器实例 |

## 对 wiki 的映射

- 升格页面：[wiki/entities/agentsdock.md](../../wiki/entities/agentsdock.md)
- 仓库归档：[sources/repos/zhengyiluo_agentsdock.md](../repos/zhengyiluo_agentsdock.md)、[sources/repos/zhengyiluo_agentsserver.md](../repos/zhengyiluo_agentsserver.md)

## 参考链接

- 产品站：<https://agentsdock.net/>
- 客户端：<https://github.com/ZhengyiLuo/AgentsDock>
- 后端：<https://github.com/ZhengyiLuo/AgentsServer>
