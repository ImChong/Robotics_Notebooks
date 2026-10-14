# OpenMuse（CopilotKit）

- **类型：** GitHub 仓库
- **仓库：** <https://github.com/CopilotKit/OpenMuse>
- **默认分支：** `main`
- **源码状态：** 已公开
- **许可证：** MIT
- **核验日期：** 2026-10-14
- **知识节点：** [OpenMuse](../../wiki/entities/copilotkit-openmuse.md)

## 仓库说明

OpenMuse 是一个跨 iOS、Android 与 Web 的个人代理应用，基于 CopilotKit React Native 与 AG-UI。它将聊天界面、持久浏览器、可选 Linux 工作区、持久任务执行、行动审阅和结构化结果整合在一个可自托管、可二次开发的应用中。

## 核心架构线索

官方 README 将系统划分为客户端、Hono + CopilotKit runtime、持久任务 worker、PGlite/PostgreSQL、Chromium browser worker、可选 Docker Linux computer，以及 Gmail/Calendar 等连接器。用户可检查 agent 执行进度和保存的结果；部分外部行动需审阅后再执行。

## 复现与边界

README 要求 Node.js 24 LTS、pnpm 11.19.0。最小本地样例不需要模型、Google 账户或 Docker；接入模型、Google OAuth、CopilotKit Intelligence、浏览器 worker 和 Linux computer 时需按功能配置相应服务和凭证。官方标注项目处于 Alpha，且验证文档区分了已实现功能与需要在线凭证的 live acceptance；Linux computer 默认关闭，需服务端配置。

MIT 许可证允许使用与修改。部署时仍应检查 OAuth 权限、外部行动审阅策略、隔离边界和凭证管理。

## 关联资料

- [OpenMuse 项目详情](../../wiki/entities/copilotkit-openmuse.md)
- [GitHub README](https://github.com/CopilotKit/OpenMuse/blob/main/README.md)
- [Agent computer 配置与限制](https://github.com/CopilotKit/OpenMuse/blob/main/docs/COMPUTER.md)
- [功能清单](https://github.com/CopilotKit/OpenMuse/blob/main/docs/FEATURES.md)
- [许可证](https://github.com/CopilotKit/OpenMuse/blob/main/LICENSE)
