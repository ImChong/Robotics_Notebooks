---
type: entity
tags:
  - physical-ai
  - aws
  - cloud-robotics
  - isaac-sim
  - isaac-lab
  - gr00t
  - cosmos
  - sim2real
status: complete
updated: 2026-10-10
project_id: aws-physical-ai-toolchain
project: https://github.com/aws-samples/sample-the-physical-ai-toolchain-on-aws
code: https://github.com/aws-samples/sample-the-physical-ai-toolchain-on-aws
related:
  - ../concepts/sim2real.md
  - ../concepts/world-action-models.md
  - ./video-prediction-policy-2.md
sources:
  - ../../sources/repos/aws-physical-ai-toolchain-on-aws.md
summary: "AWS Physical AI Toolchain 是基于 AWS 云服务与 NVIDIA Physical AI 软件栈的参考架构和样例集合，覆盖数据生成、训练、SIL 仿真、Sim-to-Real/HIL 与边缘部署。"
---

# The Physical AI Toolchain on AWS

## 一句话定义

**AWS Physical AI Toolchain** 是一组用于在 AWS 上搭建 Physical AI 开发闭环的参考架构、基础设施代码和部署样例。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| PAI | Physical AI | 与物理环境交互的 AI 系统总称 |
| SIL | Software-in-the-Loop | 在软件仿真环节验证策略和系统 |
| HIL | Hardware-in-the-Loop | 将真实硬件纳入闭环测试 |
| IaC | Infrastructure as Code | 以代码描述和部署云基础设施 |
| RL | Reinforcement Learning | 强化学习，用于策略训练 |
| VLA | Vision-Language-Action | 视觉、语言与动作联合策略 |

## 这是什么

AWS 样例仓库把 Physical AI 工程拆成一条云端工作流：生成或整理数据、训练策略、在仿真中验证，再部署到边缘设备并把运行结果反馈到下一轮开发。它将 AWS 托管服务与 NVIDIA Isaac Sim、Isaac Lab、GR00T、Cosmos 和 OSMO 等工具组合起来，目标是提供可复用的基础设施起点，而非提出一个新的机器人学习算法。

## 架构与组件

仓库用四个环节描述开发飞轮：

1. **合成数据生成：** 场景组合、域随机化和生成式数据增强。
2. **模型训练：** 模仿学习、强化学习、微调和 checkpoint 管理。
3. **SIL 仿真：** 物理验证、对抗场景与回归测试。
4. **Sim-to-Real / HIL：** 域适配、安全监控和边缘部署。

NVIDIA OSMO 被用作跨异构算力的调度与数据依赖编排层；Strands Agents 示例则探索通过自然语言编排机器人任务。仓库把各阶段拆成独立模块，已有入口包括 Foundation、Cosmos、Isaac Lab、Isaac GR00T、DreamZero、Isaac Sim、OSMO、Strands Agents 和 Isaac Lab Arena Evaluation。仓库 README 将 Edge Deployment 标为 planned，模块的状态需按对应目录核实。

## 数据到部署的闭环

```mermaid
flowchart LR
  A["遥操作与仿真数据"] --> B["S3 数据整理"]
  B --> C["GR00T 模仿学习或 Isaac Lab 强化学习"]
  C --> D["Cosmos 合成数据与 Isaac Sim SIL 验证"]
  D --> E["Jetson 或边缘设备部署"]
  E --> F["运行反馈与新数据"]
  F --> A
  O["OSMO 工作流编排"] -. "调度各阶段" .-> B
  O -. "调度各阶段" .-> C
  O -. "调度各阶段" .-> D
  O -. "调度各阶段" .-> E
```

仓库的端到端示例以 UR3 和 Robotiq 2F-85 为例，包含 27 条真实遥操作 episode；示例将遥操作数据整理成 LeRobot 格式，在云上微调 GR00T，并组合仿真、数据生成和部署步骤。换机器人时需要提供自己的模型和数据，示例工作流并不会自动解决每种本体的驱动、标定或安全问题。

## 使用时要核对的事项

- **先选模块再部署：** 已有 Kubernetes 或自建流水线时，可单独部署相关工具；要搭建完整控制面时再考虑 OSMO。
- **成本是配置相关的：** README 的估算依赖实例规格、区域、Spot/Capacity Block 与运行时间；应在当前 AWS 区域重新核算。
- **权重和容器有各自条件：** NGC、Hugging Face、模型权重、仿真资产及云服务分别可能要求账号、许可或配额。
- **不是实时控制架构：** AWS 云端负责训练、仿真和管理；机器人硬实时控制仍需要在本地控制器/边缘节点满足时序与安全约束。
- **不是自动化安全认证：** HIL 和回归门禁是工程流程组件，不等于通过了特定机器人或行业的安全认证。

## 源码运行时序图

入口：Dockerfile 中生成的 `/opt/ml/code/entrypoint.sh`（容器 ENTRYPOINT；SageMaker 无参数执行 `train_entrypoint.py`，AWS Batch 以 `batch-train` 参数执行 `batch-train-entrypoint.sh`）。下图只覆盖仓库中 GR00T N1.6 微调容器的运行路径，不代表整个工具链。

```mermaid
sequenceDiagram
    autonumber
    participant EP as entrypoint.sh Dockerfile 生成的容器入口
    participant BT as batch-train-entrypoint.sh Batch 分支
    participant TE as train_entrypoint.py
    participant GS as Isaac-GR00T SDK experiment.run
    participant S3 as S3

    alt SageMaker 模式 无参数
        EP->>TE: exec python train_entrypoint.py
    else AWS Batch 模式 command 为 batch-train
        EP->>BT: exec bash batch-train
        BT->>S3: aws s3 sync DATASET_S3_URI 到数据目录
        BT->>TE: python3 train_entrypoint.py 读取 SM_HP_* 环境变量
    end
    TE->>TE: require_sdk 检查 SDK 目录与 import torch
    TE->>TE: maybe_relaunch_with_torchrun 多卡时预下载模型后 torch.distributed.run 重新执行自身
    Note over TE: 子进程重新进入 __main__ 后各 rank 都执行 run_training
    TE->>TE: run_training 调用 _hp 读取 hyperparameters.json 或 SM_HP_*
    TE->>TE: write_embodiment_config 写出 ur3_embodiment/ur3_config.py
    TE->>TE: load_modality_config 通过 importlib 注册 NEW_EMBODIMENT
    TE->>GS: get_default_config 后 load_dict 并设置 model training data 参数
    TE->>TE: 删除过期的 stats 文件
    TE->>GS: run config 开始微调
    GS-->>TE: 返回 checkpoint 写入 output_dir
    opt rank 0 且 main 继续执行
        TE->>TE: write_baselines_only 写 eval_report.json 仅为数据集基线
        TE->>TE: 写 training_metadata.json
    end
    opt Batch 且设置了 CHECKPOINT_BUCKET
        BT->>S3: aws s3 sync 输出目录到 checkpoint 前缀
    end
```

图依据：aws-samples/sample-the-physical-ai-toolchain-on-aws@7156dd4，实际读过 `README.md`、`isaac-gr00t-on-aws/README.md`、`containers/gr00t-training/train_entrypoint.py`、`containers/gr00t-training/batch-train-entrypoint.sh`、`containers/gr00t-training/Dockerfile`（COPY、ENTRYPOINT 段）。

## 关联页面

- [仿真到真机迁移](../concepts/sim2real.md)
- [世界-动作模型](../concepts/world-action-models.md)
- [Video Prediction Policy 2（VPP2）](./video-prediction-policy-2.md)

## 参考来源

- [AWS 官方仓库来源归档](../../sources/repos/aws-physical-ai-toolchain-on-aws.md)
- [sample-the-physical-ai-toolchain-on-aws](https://github.com/aws-samples/sample-the-physical-ai-toolchain-on-aws)
- 源码核对（2026-11-02）：aws-samples/sample-the-physical-ai-toolchain-on-aws@7156dd4，读过 README.md、isaac-gr00t-on-aws/README.md、containers/gr00t-training/train_entrypoint.py、batch-train-entrypoint.sh、Dockerfile（COPY 与 ENTRYPOINT 段）；时序图依据以上文件，未运行代码。
