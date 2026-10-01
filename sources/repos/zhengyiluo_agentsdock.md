# ZhengyiLuo/AgentsDock

> 来源归档

- **标题：** AgentsDock
- **类型：** repo（桌面 + 移动客户端 monorepo）
- **作者：** Zhengyi Luo（GitHub）
- **代码：** <https://github.com/ZhengyiLuo/AgentsDock>
- **项目页：** <https://agentsdock.net/>
- **许可：** Apache-2.0（以仓库 `LICENSE` 为准）
- **入库日期：** 2026-10-01
- **一句话说明：** 面向 agentic AI 研究的跨端客户端：Electron 桌面（`electron/`）+ Expo/React Native 移动（`mobile-react/`）+ 产品站（`website/`）；通过自托管 [AgentsServer](zhengyiluo_agentsserver.md) 驱动本机已安装的 Claude Code / Codex / Cursor / OpenCode 等 CLI。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [AgentsDock](../../wiki/entities/agentsdock.md) | 实体页：架构、开源边界、与 Hermes / Agent Reach 对照 |
| [Zhengyi Luo](../../wiki/entities/zhengyi-luo.md) | 维护者节点 |
| [AgentsDock 产品站](../sites/agentsdock-net.md) | 步骤 2.5 项目页归档 |

## 仓库结构（README 归纳）

| 目录 | 用途 |
|------|------|
| `electron/` | 当前 macOS / Linux / Windows 桌面客户端 |
| `mobile-react/` | iOS / Android 客户端与原生模块 |
| `website/` | agentsdock.net 静态站 |
| `server/` | 与 AgentsServer 同步的服务端运行时（迁移期与独立后端仓并存） |
| `team-hub/` | Team Hub 服务与测试 |
| `docs/` | 架构、发布通道、迁移说明 |

## 开发要点

- 工具链：**Node.js 24**、**pnpm 11.9.0**（pin）；本地开发需已运行的 AgentsServer。
- Stable 桌面发布见 [GitHub Releases](https://github.com/ZhengyiLuo/AgentsDock/releases)；CI 验证源码，**不**在 CI 内发布安装包。

## 对 wiki 的映射

- 与 [sources/repos/zhengyiluo_agentsserver.md](zhengyiluo_agentsserver.md)、[sources/sites/agentsdock-net.md](../sites/agentsdock-net.md) 三处互指；实体 **[wiki/entities/agentsdock.md](../../wiki/entities/agentsdock.md)**。
