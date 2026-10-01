# ZhengyiLuo/AgentsServer

> 来源归档

- **标题：** AgentsServer
- **类型：** repo（自托管 Python 后端）
- **作者：** Zhengyi Luo（GitHub）
- **代码：** <https://github.com/ZhengyiLuo/AgentsServer>
- **客户端 / 项目页：** <https://agentsdock.net/> · [AgentsDock](zhengyiluo_agentsdock.md)
- **许可：** Apache-2.0（以仓库 `LICENSE` 为准）
- **入库日期：** 2026-10-01
- **一句话说明：** AgentsDock 的 **自托管执行后端**：在拥有项目与 agent CLI 的机器上持久化聊天、调度代理回合、暴露文件/终端/定时任务等能力；默认 HTTP 端口 **7850**，由 `./install.sh`（依赖 [`uv`](https://docs.astral.sh/uv/)）引导安装。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [AgentsDock](../../wiki/entities/agentsdock.md) | 实体页：客户端–服务器分工与部署读法 |
| [AgentsDock 产品站](../sites/agentsdock-net.md) | 连接与远程访问（Tailscale 等）文档入口 |

## 运行与运维要点（README 归纳）

- **主机**：Linux 或 Apple Silicon macOS；代理 CLI 须在 **服务器** 安装并完成提供商认证。
- **内存门禁**：新 agent 回合默认要求 **≥2 GiB 可用 RAM**，定时任务 **≥4 GiB**（可配置，见仓库 `docs/MEMORY_ADMISSION.md`）。
- **多实例**：`./instances.sh new` 可在单机起多个独立 server（不同端口/名称）。
- **更新**：`git pull --ff-only && ./install.sh` 或客户端 Settings 内签名更新（以官方 update 文档为准）。
- **终端**：可选依赖 `tmux` 以提供持久 shell。

## 开源边界

- **已开源**：服务端 Python 代码、安装/卸载脚本、文档与发布包。
- **非本地**：LLM 推理仍由 Claude / OpenAI Codex / Cursor 等 **用户配置的提供商** 处理；自托管指的是 **数据与 CLI 执行面**，不是权重或模型私有化。

## 对 wiki 的映射

- 与 [sources/repos/zhengyiluo_agentsdock.md](zhengyiluo_agentsdock.md)、[sources/sites/agentsdock-net.md](../sites/agentsdock-net.md) 互指；实体 **[wiki/entities/agentsdock.md](../../wiki/entities/agentsdock.md)**。
