# OpenBot（CopilotKit）官方代码与架构归档

> 来源归档（repo / 官方文档；不是 Intel Labs 同名轮式机器人 OpenBot）

- **标题：** OpenBot — AI coworkers, ready for the enterprise / 可自托管 AI 同事工作空间
- **作者 / 维护方：** CopilotKit
- **类型：** repo（应用模板与配套文档）
- **仓库：** <https://github.com/CopilotKit/OpenBot>
- **官网：** <https://open.bot/>
- **官方架构：** <https://github.com/CopilotKit/OpenBot/blob/main/docs/architecture.md>
- **工作机制：** <https://github.com/CopilotKit/OpenBot/blob/main/docs/how-openbot-works.md>
- **Bot 接入：** <https://github.com/CopilotKit/OpenBot/blob/main/docs/coworkers.md>
- **源码状态：** 模板与前端等代码已公开，MIT；API server、agent-computer 等运行时通过发布的 Docker 镜像使用，不能据模板源码认定全部运行时源代码均在此仓库内
- **许可边界：** 仓库应用模板 MIT；CopilotKit Intelligence 是独立许可组件，团队生产部署可能需要另行授权
- **入库日期：** 2026-11-17
- **核对范围：** 官方 README 的架构、功能、运行、配置、API、许可章节；docs/architecture.md 的 Runtime flow、Browser action governance、Computers、MCP and skills、Security boundaries 关键章节；docs/how-openbot-works.md 与 docs/coworkers.md 的 Bot/工具/人工审批机制；package.json、scripts/start.sh 启动代码；官方官网及许可说明。未拉取发布的容器镜像、未运行端到端部署，未验证各云服务和第三方模型的实际可用性。
- **通读：** 已完成
- **知识节点：** [CopilotKit OpenBot](../../wiki/entities/copilotkit-openbot.md)
- **官网归档：** [open.bot 官方页面](../sites/copilotkit-openbot-project.md)

## 原文核验要点

1. **工作空间定位：** OpenBot 提供 Bot 角色、会话、定时任务、记忆、技能、应用/工具、Computer、审批与活动审计；不是一个固定的底层智能体算法，也不提供机器人运动控制栈。
2. **三层架构：** 前端 React 客户端；Hono API、Agent 管理/策略网关、PostgreSQL、计算机 Supervisor；独立 CopilotKit Intelligence 服务保管持久线程和相关能力。外部 Agent 用 AG-UI 接入。
3. **工具治理：** Browser / File / MCP 工具经服务端做目标解析、策略评估与审计；策略可按 bot、actor、网页、命令、MCP 工具和发起者过滤。不同角色的授权需单独配置。
4. **计算资源：** Agent computer 含 Chromium、工作空间与终端；Supervisor 能按人或隔离 Bot 管理容器。普通 Docker 共享容器模式不等于硬隔离；网络策略与镜像/凭证边界需独立验证。
5. **模型与部署：** 支持配置 OpenAI、Anthropic、Google 或 OpenAI 兼容服务；README 提供 macOS / Windows 安装入口、Docker Compose 源码模板、Docker 单容器与 Kubernetes Chart；部分部署流程需要 CopilotKit Intelligence 和模型密钥。
6. **开源边界：** README 明确“application template” MIT，与 separately licensed Intelligence 分开；`scripts/start.sh` 会从 `backend-images.env` 对应 tag 拉取后端、计算机与 Supervisor 镜像。公开模板可修改，但无法仅依这一仓库复核镜像内部实现。

## 复现入口与执行路径

- `cp .env.example .env` → `bun install` → 配置 Intelligence 及模型密钥 → `bash scripts/start.sh`，前端默认 `localhost:3010`，API 端口默认 `3001`。
- `scripts/start.sh` 校验 bash / 环境配置，选择镜像 tag、端口与个人版/团队版 Bot endpoint，拉起 PostgreSQL、server、computer 等服务并做健康检查。
- 按 `docs/architecture.md#runtime-flow`：前端建立会话 → server 解析角色/授权 → CopilotKit runtime 调 AG-UI Bot endpoint → 网关治理工具调用 → 允许后调用 computer/MCP → audit 和 Intelligence 线程保存事件。
- 文档演示的是编排与治理能力，不构成任意机器人机械臂、Nav2 或 ROS 2 接口已被官方支持的证据。

## 与同名/相邻项目区分

- [Intel Labs OpenBot](https://github.com/ob-f/OpenBot)：使用 Android 手机作为轮式机器人算力与视觉核心，是另一项目。
- [CopilotKit OpenMuse](https://github.com/CopilotKit/OpenMuse)：同一组织的另一开源个人 Agent 产品，仓库身份及部署结构与 OpenBot 不同。

## 官方资料

- [README 与源码](https://github.com/CopilotKit/OpenBot)
- [Architecture](https://github.com/CopilotKit/OpenBot/blob/main/docs/architecture.md)
- [Coworkers / AG-UI registration](https://github.com/CopilotKit/OpenBot/blob/main/docs/coworkers.md)
- [Approvals](https://github.com/CopilotKit/OpenBot/blob/main/docs/approvals.md)
- [Starting script](https://github.com/CopilotKit/OpenBot/blob/main/scripts/start.sh)
- [License](https://github.com/CopilotKit/OpenBot/blob/main/LICENSE)
