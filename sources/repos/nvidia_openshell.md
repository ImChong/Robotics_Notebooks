# NVIDIA/OpenShell

> 来源归档

- **标题：** OpenShell — safe runtime for autonomous AI agents
- **类型：** repo
- **组织：** NVIDIA
- **代码：** <https://github.com/NVIDIA/OpenShell>
- **文档：** <https://docs.nvidia.com/openshell/latest/>
- **许可：** Apache License 2.0
- **Stars：** ~8k+（2026-09-29，随时间变化）
- **入库日期：** 2026-09-29
- **一句话说明：** **Agent-first 开源运行时**：声明式 YAML 策略 + 内核级沙箱，限制文件/进程/网络/推理路由；Gateway 管理多沙箱生命周期，Supervisor 在 agent 进程外执行 L7 策略与 OCSF 审计。
- **沉淀到 wiki：** [`wiki/entities/nvidia-openshell.md`](../../wiki/entities/nvidia-openshell.md)
- **关联博客：** [`nvidia_open_agent_safety_platform_2026-09-28.md`](../blogs/nvidia_open_agent_safety_platform_2026-09-28.md)、<https://developer.nvidia.com/blog/add-runtime-controls-to-ai-agents-with-nvidia-openshell/>

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（Apache-2.0；仓内 LICENSE） |
| **代码** | <https://github.com/NVIDIA/OpenShell> |
| **定位** | **运行时 / 控制平面** — 包装 Claude Code、Codex、OpenCode、Copilot CLI 等，不要求改 agent 源码 |
| **非本仓范围** | **NVIDIA Sentry**、BlueField-4 硅内执行属于 [Open Agent Safety Platform](../sites/nvidia-open-agent-safety-platform-developer.md) 硬件层，非 OpenShell GitHub 交付物 |

## README / 文档要点（2026-09-29）

- **双层治理：** 内核 instrumentation 强制 syscall/文件/网络；**formal verification** 在策略变更批准前分析新增权限风险。
- **策略维度：** Filesystem / Network（可热更新 `openshell policy set|update`）/ Process / Inference（模型 API 路由到受控后端）。
- **凭证：** **Providers** — API key 等经注入环境变量，**不写入沙箱文件系统**。
- **默认网络：** default-deny；L7 可区分同 API 的 read vs write（GET vs POST）。
- **审计：** 策略决策记录为 **OCSF**；拦截时可返回描述性错误供 agent 自修正。
- **安装：** `curl -LsSf .../install.sh | sh` 或 `uv tool install -U openshell`；依赖可达 Gateway + Docker/K8s/Podman/MicroVM 等 compute driver。
- **云试跑：** Brev Launchable（README 链接 `env-3Ap3tL55zq4a8kew1AuW0FpSLsg`）。

## 对 wiki 的映射

- [NVIDIA OpenShell](../../wiki/entities/nvidia-openshell.md)
- [NVIDIA Open Agent Safety Platform](../../wiki/entities/nvidia-open-agent-safety-platform.md)
