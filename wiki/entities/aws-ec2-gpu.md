---
type: entity
tags: [infrastructure, gpu-cloud, training, international, aws, enterprise]
status: complete
updated: 2026-07-02
related:
  - ./google-cloud-gpu.md
  - ./lambda-cloud.md
  - ../comparisons/international-gpu-cloud-platforms.md
  - ./isaac-lab.md
sources:
  - ../../sources/sites/aws-ec2-gpu.md
summary: "AWS EC2 GPU 实例（g4dn/g5/p4d/p5 等）按秒计费，可与 Spot、Reserved 及 SageMaker 集成；适合数据与合规已在 AWS 的企业级训练与推理。"
---

# AWS EC2 GPU

**Amazon EC2 GPU 实例**是 AWS 弹性计算中的 **NVIDIA GPU 虚拟机**产品线，从入门级 **g4dn（T4）** 到 **p5（H100）** / **p5e（H200）**，覆盖推理至大规模训练。

## 一句话定义

当数据集、权限与流水线已在 **S3/IAM/VPC** 内时，应在同区域租 EC2 GPU 或 SageMaker，而不是把数据反复迁到 neocloud 省每小时几美元却付 egress。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| EC2 | Elastic Compute Cloud | AWS 虚拟机服务 |
| GPU | Graphics Processing Unit | g/p 实例族加速器 |
| EFA | Elastic Fabric Adapter | 多机训练高速网络 |
| Spot | EC2 Spot Instance | 可中断低价实例 |
| SageMaker | Amazon SageMaker | 托管 ML 训练/推理 |
| S3 | Simple Storage Service | 常见数据集存放 |

## 为什么重要

- **企业默认云**：机器人公司数据湖、标注与 MLOps 常在 AWS。
- **多节点成熟**：p4d/p5 + EFA；Capacity Blocks 预订 H100 窗口。
- **本库 legacy 入口**：训练资源列表曾列 AWS。

## 核心结构 / 机制

| 实例族 | GPU | 场景 |
|--------|-----|------|
| g4dn | T4 | 轻量推理 |
| g5 | A10G | 中等训练 |
| p4d/p4de | A100 | 多卡训练 |
| p5/p5e | H100/H200 | 前沿大模型/大规模 RL |

- **计费**：On-Demand / Spot / Reserved / Savings Plans
- **运维**：自选 AMI、驱动、集群软件（或用 SageMaker 减负）

## 常见误区或局限

- **标价远高于 neocloud**：p5 8×H100 按需可达数十美元/小时量级。
- **自己管环境**：除非用 Deep Learning AMI 或 SageMaker。
- **Spot 可中断**：长跑需 checkpoint（与 Vast 类似逻辑）。

## 与其他页面的关系

- [国外 GPU 云平台选型](../comparisons/international-gpu-cloud-platforms.md)
- [Google Cloud GPU](./google-cloud-gpu.md) — 超大规模云对照
- [Lambda Cloud](./lambda-cloud.md) — 更低价的专用 AI 云

## 推荐继续阅读

- [EC2 实例类型](https://aws.amazon.com/ec2/instance-types/)
- [P5 实例（H100）](https://aws.amazon.com/ec2/instance-types/p5/)

## 参考来源

- [AWS EC2 GPU 归档](../../sources/sites/aws-ec2-gpu.md)
