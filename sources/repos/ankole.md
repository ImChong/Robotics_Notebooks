# Ankole（AgentBull/ankole）官方仓库与源码阅读

> 来源归档（repo）· 开源企业 Agent Harness / 多智能体协作平台

- **标题：** Ankole — Open-source Pod workspace and multi-agent runtime / 开源企业数字总部与多智能体运行时
- **组织：** AgentBull
- **类型：** repo
- **代码：** https://github.com/AgentBull/ankole
- **官网：** https://ankole.agentbull.com/
- **官方中文文档：** https://ankole.agentbull.com/zh-Hans-CN/docs/introduction/
- **许可：** Apache-2.0（根目录 LICENSE，README 双语说明）
- **技术栈：** Elixir / Phoenix / OTP（控制面）、Bun + TypeScript（Agent Computer）、Rust（Kernel）、PostgreSQL 18+（持久化）；Docker Compose / Helm
- **入库日期：** 2026-11-18
- **核对范围：** 完整阅读默认分支 README.zh-Hans.md（233 行、快速开始、Pod 模块、典型闭环、架构、限制和许可证）；对读 README.md、package.json、app/agent_computer/README.md、docs/TradeoffsAndKnownLimits.md 与 docs/design-docs/Signals.md、Sessions.md、AgentWorkers.md 的主要段落；检查 app/control_plane/lib/ankole/signals.ex、sessions.ex、app/agent_computer/src/main.ts、worker/active-turns.ts、worker/turn-execution.ts 的调用边界。部分长设计文档及其他源码未逐行通读；未实机部署。
- **通读：** 已完成
- **沉淀到 wiki：** [Ankole 项目实体页](../../wiki/entities/ankole.md)

## 一句话说明

Ankole 把可协作的文档 / Base / Applet / Pod Chat、个人助理和职能 AI 同事放进同一 Pod；通过企业大脑、外部连接器、权限与受控 Agent Computer Worker 承载长期有人审批的业务闭环。它是企业级通用 Agent 平台，不是机器人运动控制库。

## 核心架构：对齐官方 README 与源码

| 模块 | 职责 | 官方核对入口 |
|---|---|---|
| Pod | Documents、Bases、Applets、Slides、Pod Chat 与成员权限 | README.zh-Hans.md 的「Pod 在线办公室的五大模块」 |
| Meta Ankole | Company Brain 知识图、Skills、AGENTS.md / DESIGN.md 契约 | README.zh-Hans.md 的「企业大脑」 |
| Control Plane | Elixir/Phoenix/OTP：授权、会话、Signals、Driver、持久状态与 AIGateway | app/control_plane/lib/ankole/sessions.ex、signals.ex；docs/design-docs/Sessions.md |
| Function Runtime | 无状态执行小应用后端与确定性 Applet Actions | README.zh-Hans.md 的「工作原理」 |
| Agent Computer Worker | 每个 Agent 独立 Worker，Bun/TS、Codex、浏览器、MCP、Skills、/drive | app/agent_computer/README.md、src/main.ts |
| Kernel | Rust 密码学、权限求值、编解码等底层能力 | README.zh-Hans.md 的「工作原理」 |
| 数据持久化 | PostgreSQL、Drive 对象存储、每个 Agent 的 /workspace | docs/TradeoffsAndKnownLimits.md |

## 开源状态与部署

- **开源结论：** 官方公开仓库、Docker Compose / Helm 配置及源码，Apache-2.0；模型提供商与 SaaS 凭据需要自行接入。
- **最短部署路径（按官方 README，未执行）：** 克隆仓库 → 在 tools/deploy/docker-compose 复制 .env.example → 设置域名、数据库密码、三个不同的随机密钥及 Docker GID → docker compose pull / up -d → 从 control-plane 日志读取一次性激活码 → HTTPS /setup 配置 Email OTP 或 SSO、LLM Provider、Agent。
- **基础要求：** Docker 宿主机、PostgreSQL 与存储空间、每个活跃 Agent 的独立 Worker；Docker socket 挂载相当于高权限宿主机访问，应使用受信任的部署环境。
- **支持集成：** 飞书、钉钉、企业微信、Slack、Teams 等 IM，GitHub、Linear、email 与 MCP；不同通道的外网 Webhook 需求不同，详见官方指南。
- **版本边界：** 官方 README 称已用于生产，但公开 API 不承诺跨版本兼容；不能把“可自托管”当作“已做独立安全审计”。

## 有源码依据的执行路径

1. `Ankole.Signals` 通过持久化 Signal、事务与 EnsureDriver 唤醒处理输入；`Ankole.Sessions` 中的 `ensure_driver` 确保同一 Session Driver。
2. 会话/模型历史以 PostgreSQL 作为恢复真相；`Sessions.md` 明确 AIGateway 负责模型输入输出，Conversations 负责公开消息。
3. 需电脑的 Background Agent Job 经 RuntimeFabric 路由到对应 Worker；`src/main.ts` 建立连接并挂载 Worker 的 `ActiveTurns` 和工作空间。
4. `worker/active-turns.ts` 接收 turn、检验容量、发回接受/失败；`worker/turn-execution.ts` 创建 Drive 和工作目录，再调用 `runCodexJob`。注意 **这是 Codex Job 运行路径，不代表所有普通聊天轮次都走 Codex**。

## 已知限制与机器人场景边界

- 单站点多租户在逻辑上隔离，但共享站点运维方、控制面、Function Runtime 与基础设施；**不是对互不信任租户的强计算隔离**。
- 同一 Agent 的 Sessions/Jobs 共享 `/workspace`；对不同 Agent 的 Worker 隔离，不等于 Agent 内部每个会话互相隔离。
- 云 Worker 后端的存储保证有区别，Cloudflare 快照模式的异常中断可能丢失最近本地写入；必须验证备份、恢复和单写者 fencing。
- 没有证据证明 Ankole 内置 ROS2、实时运动控制闭环、硬件急停或机器人运动安全认证。集成具身系统时应只用作上层任务编排，机器人侧实时控制与安全必须独立实现。

## 对 wiki 的映射

- [Ankole 独立实体](../../wiki/entities/ankole.md)：企业多智能体运行层、Company Brain 与安全边界。
- [Hermes Agent](../../wiki/entities/hermes-agent.md)：同属长期 Agent 运行时；Hermes 偏个人常驻助理，Ankole 偏企业 Pod、身份治理和结构化业务数据。
- [Agnes Harness](../../wiki/entities/agnes-harness.md)：可对照 agent harness 与设备任务接口的边界。
- [LLM 机器人控制接口](../../wiki/concepts/llm-robotics-control-interfaces.md)：上层任务级 Agent 与设备侧实时控制的分工。

## 官方入口

- 仓库（README、中英文）：https://github.com/AgentBull/ankole
- Docker Compose：https://github.com/AgentBull/ankole/tree/main/tools/deploy/docker-compose
- 设计约束：https://github.com/AgentBull/ankole/blob/main/docs/TradeoffsAndKnownLimits.md
- Sessions：https://github.com/AgentBull/ankole/blob/main/docs/design-docs/Sessions.md
- Agent Workers：https://github.com/AgentBull/ankole/blob/main/docs/design-docs/AgentWorkers.md
