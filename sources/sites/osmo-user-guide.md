# OSMO User Guide（文档索引）

> 来源归档

- **标题：** User Guide — OSMO Documentation
- **类型：** site（NVIDIA 官方文档）
- **链接：** <https://nvidia.github.io/OSMO/main/user_guide/index.html>
- **仓库文档根：** <https://nvidia.github.io/OSMO/main/>
- **代码：** <https://github.com/NVIDIA/OSMO>
- **入库日期：** 2026-09-06
- **文档版本线：** main / release/6.0–6.4（页面顶栏，2026-09-06 可见）
- **一句话说明：** OSMO **开源 Physical AI 工作流编排**用户指南：YAML 定义训练/仿真/边缘任务；解决 **Three Computer Problem**；支持 EKS/AKS/GKE/on-prem/air-gapped。
- **沉淀到 wiki：** [`wiki/entities/nvidia-osmo.md`](../../wiki/entities/nvidia-osmo.md)

## Overview 摘录

- **声明式 YAML：** 整条 Physical AI 管线（训练、仿真、HIL）写在一个 workflow 文件；OSMO 管理依赖与资源分配。
- **Three Computer Problem：**
  - 🧠 **Training GPUs** — GB200、H100（DL/RL）
  - 🌐 **Simulation** — RTX PRO 6000（物理与传感器渲染）
  - 🤖 **Edge** — Jetson AGX Thor（HIL 与验证）
- **工作流四步：** Define（YAML）→ Submit（CLI/Web UI）→ Execute（调度依赖）→ Iterate（对象存储输出 + 监控）。

## Why Choose OSMO（文档表）

| 能力 | 说明 |
|------|------|
| Zero-Code Orchestration | YAML 定义任务，无 Python glue |
| Group Scheduling | 训练+仿真+边缘测试 **同一 workflow 并行/串行** |
| Truly Portable | 笔记本/云/on-prem 同一文件 |
| Smart Storage | task output 或 object storage 传数据 |
| Interactive Development | VSCode/Jupyter/SSH 连运行中任务 |
| Infrastructure-Agnostic | workflow 不绑定具体集群实现 |

## Key Benefits 教程映射

| 场景 | 文档教程 |
|------|----------|
| 远程 GPU 交互开发 | Interactive Workflows |
| Isaac Sim 规模化 SDG | Isaac Sim SDG |
| 分布式训练 | Model Training |
| 数据并行 RL | Reinforcement Learning |
| 仿真 + HIL 验证 | Hardware In The Loop |
| 数据变换与后处理 | Working with Data |
| Jetson/真机 benchmark | Hardware Testing |

## Bring Your Own Infrastructure

- **Compute：** 任意 K8s 集群 — AWS EKS、Azure AKS、Google GKE、on-prem、**Jetson 嵌入式**。
- **Storage：** S3 兼容或 Azure Blob；artifact 跨 backend 共享。

## 对 wiki 的映射

- 实体：[`wiki/entities/nvidia-osmo.md`](../../wiki/entities/nvidia-osmo.md)
- 产品页：[`sources/sites/nvidia-osmo-developer.md`](./nvidia-osmo-developer.md)
- 仓库：[`sources/repos/nvidia_osmo.md`](../repos/nvidia_osmo.md)
