# NVIDIA Open Agent Safety Platform（Developer 门户）

> 来源归档（site / NVIDIA Developer）

- **标题：** NVIDIA Open Agent Safety Platform
- **类型：** site
- **原始链接：** https://developer.nvidia.com/blog/nvidia-open-agent-safety-platform-a-reference-for-continuous-in-silicon-agent-monitoring/（首发叙事见 Developer Blog）
- **OpenShell 入口：** https://docs.nvidia.com/openshell/latest/
- **OpenShell 代码：** https://github.com/NVIDIA/OpenShell
- **入库日期：** 2026-09-29
- **一句话说明：** NVIDIA 提出的 **agent 安全参考设计**：软件层 **OpenShell** + 硬件层 **Sentry on BlueField-4**，面向 Vera Rubin POD 等 AI factory 规模的带外线速监控。

## 源码与组件开放核查（步骤 2.5）

| 组件 | 开放程度 | 说明 |
|------|----------|------|
| **NVIDIA OpenShell** | **已开源** | GitHub `NVIDIA/OpenShell`，Apache 2.0 |
| **NVIDIA Sentry** | **部分 / 产品化** | 博客描述为 BlueField-4 上可选安全层，经 DOCA 与 OpenShell 策略联动；**无与 OpenShell 同级的单一公开应用仓**（截至 2026-09-29 入库日，以官方文档与博客为准） |
| **BlueField-4 / Vera** | **硬件平台** | 策略在硅内/ DPU 路径执行依赖对应系统栈与软件更新 |

## 关联归档

- [博客摘录](../blogs/nvidia_open_agent_safety_platform_2026-09-28.md)
- [OpenShell 仓库](../repos/nvidia_openshell.md)

## 对 wiki 的映射

- [NVIDIA Open Agent Safety Platform](../../wiki/entities/nvidia-open-agent-safety-platform.md)
- [NVIDIA OpenShell](../../wiki/entities/nvidia-openshell.md)
