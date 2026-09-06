# NVIDIA/OSMO

> 来源归档

- **标题：** OSMO — Workflow Orchestration for Physical AI
- **类型：** repo
- **组织：** NVIDIA
- **代码：** <https://github.com/NVIDIA/OSMO>
- **文档：** <https://nvidia.github.io/OSMO/main/user_guide/index.html>
- **产品页：** <https://developer.nvidia.com/osmo>
- **Stars：** ~218（2026-09-06）
- **入库日期：** 2026-09-06
- **一句话说明：** **Physical AI 专用 YAML 工作流编排器**：在异构 Kubernetes 上统一调度 **训练 GPU（GB200/H100）**、**仿真 GPU（RTX PRO）** 与 **边缘 Jetson Thor HIL**，CLI + Agent context 驱动端到端管线。
- **沉淀到 wiki：** [`wiki/entities/nvidia-osmo.md`](../../wiki/entities/nvidia-osmo.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（Apache-2.0；见仓内 LICENSE） |
| **代码** | <https://github.com/NVIDIA/OSMO> |
| **定位** | **编排层** — 不替代 Isaac Sim / PyTorch / SLURM 本身；不直接部署量产机器人 |
| **生产背书** | README 称已在 NVIDIA 内部支撑 GR00T、Isaac Lab、Isaac Sim、Isaac ROS 等 workloads |

## README 要点（2026-09-06）

- **Three Computer Problem：** 训练（云 GB200/H100）+ 仿真（RTX PRO 物理/传感器渲染）+ 边缘（Jetson AGX Thor HIL）— 单 YAML 串联。
- **零代码 YAML：** `workflow.tasks` 定义镜像、平台、GPU 资源、`inputs`/`outputs`（含 S3）。
- **可移植：** 笔记本 Docker/KIND → EKS/AKS/GKE/本地/on-prem/air-gapped，**零改 workflow**。
- **交互开发：** 远程任务上开 VSCode / Jupyter / SSH。
- **平台工程：** 集中控制面、动态注册 K8s backend、跨云/区域/边缘池化 GPU。
- **教程链：** Interactive Workflows、Isaac Sim SDG、Model Training、RL、HIL、Hardware Testing、Working with Data。
- **Cookbook：** 仓内 `cookbook/` 机器人 workflow 示例。
- **Roadmap（摘录）：** Q1 2026 OAuth/Okta/Azure AD、Marketplace 一键部署；长期 Python workflow API、负载感知多 backend、动态扩缩 workflow。

## 示例 workflow（README 摘录）

```yaml
workflow:
  tasks:
  - name: simulation
    image: nvcr.io/nvidia/isaac-sim
    platform: rtx-pro-6000
  - name: train-policy
    image: nvcr.io/nvidia/pytorch
    platform: gb200
    resources:
      gpu: 8
    inputs:
    - task: simulation
  - name: evaluate-thor
    image: my-ros-app
    platform: jetson-agx-thor
    inputs:
    - task: train-policy
    outputs:
    - url: s3://my-bucket/thor-benchmark/
```

## 对 wiki 的映射

- 实体：[`wiki/entities/nvidia-osmo.md`](../../wiki/entities/nvidia-osmo.md)
- 产品页：[`sources/sites/nvidia-osmo-developer.md`](../sites/nvidia-osmo-developer.md)
- User Guide：[`sources/sites/osmo-user-guide.md`](../sites/osmo-user-guide.md)
