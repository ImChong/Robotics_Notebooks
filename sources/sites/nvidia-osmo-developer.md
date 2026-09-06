# NVIDIA OSMO（developer.nvidia.com）

> 来源归档

- **标题：** NVIDIA OSMO
- **类型：** site（NVIDIA 官方产品/开发者门户）
- **链接：** <https://developer.nvidia.com/osmo>
- **代码：** <https://github.com/NVIDIA/OSMO>
- **文档：** <https://nvidia.github.io/OSMO/main/user_guide/index.html>
- **入库日期：** 2026-09-06
- **一句话说明：** **开源、Agentic 的 Physical AI 编排器**：YAML 定义训练/仿真/边缘 HIL 全链；CLI + Agent context 让编码 Agent 查询 workflow、GPU 容量与平台状态；**不是 MLOps 平台**，专注执行与数据血缘。
- **沉淀到 wiki：** [`wiki/entities/nvidia-osmo.md`](../../wiki/entities/nvidia-osmo.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源** — Deploy 按钮链 GitHub |
| **Agentic** | 交付 CLI + **agent context file**，供 Cursor/Copilot 等理解 pipeline 与集群状态 |
| **非目标** | FAQ 明确：**不**替代仿真器/训练框架；**不**直接部署量产机器人；**不是**实验看板/artifact registry 型 MLOps |

## 产品叙事摘录

- **Prompt → running pipeline：** 合成数据、训练、SIL/HIL 评测，无需自建 K8s 专家栈。
- **Centralized control plane：** x86/Arm/NVIDIA GPU 的 K8s 集群；on-prem + AWS/Azure/GCP。
- **Secure open standards：** OIDC 认证、账户、registry、storage、secrets。

## FAQ 要点（2026-09-06）

| 问题 | 结论 |
|------|------|
| 用途 | 多阶段 Physical AI workflow：数据生成、训练、仿真、评测、HIL |
| vs 仿真/训练栈 | **编排** Isaac Sim、PyTorch、RL 框架，不替换 |
| 量产部署 | 产出 policy/数据集/artifact；量产 runtime 需用户集成 |
| vs MLOps | 无 experiment dashboard；聚焦 **workflow 执行、数据集版本、数据血缘、算力调度** |
| 运行环境 | on-prem、多云、**Jetson/ARM 边缘**、混合算力 |
| vs SLURM | SLURM 是通用 HPC 调度；OSMO 面向 **数据集管理 + 仿真集成 + 异构硬件 + 多阶段机器人管线** |
| K8s  expertise | **不需要**写 manifest；YAML 抽象底层基础设施 |

## 对 wiki 的映射

- 实体：[`wiki/entities/nvidia-osmo.md`](../../wiki/entities/nvidia-osmo.md)
- 仓库：[`sources/repos/nvidia_osmo.md`](../repos/nvidia_osmo.md)
- User Guide：[`sources/sites/osmo-user-guide.md`](./osmo-user-guide.md)
