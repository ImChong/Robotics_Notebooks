# AWS EC2 GPU

> 来源归档

- **标题：** Amazon EC2 GPU Instances
- **类型：** site（超大规模云 GPU 算力）
- **来源：** Amazon Web Services
- **链接：** https://aws.amazon.com/ec2/instance-types/ 、https://aws.amazon.com/ec2/instance-types/p5/
- **入库日期：** 2026-07-02
- **一句话说明：** AWS 弹性计算 GPU 实例族：从 g4dn（T4）到 p5（H100）、p5e（H200）；按秒计费，可与 Spot/Reserved/SageMaker 集成，适合已有 AWS 数据栈的企业级训练与推理。

## 为什么值得保留

- **本库历史入口**：`sources/train.md` 与 legacy 资源地图已列 AWS。
- **企业默认**：数据在 S3、IAM 合规、多区域时，算力应同区部署避免 egress。
- **多节点**：p4d/p5 支持 EFA 高速网络；Capacity Blocks 可预订 H100 集群窗口。

## 平台要点（公开资料 2026-07 归纳）

| 实例族 | GPU | 典型用途 |
|--------|-----|----------|
| g4dn | T4 | 推理、小模型 |
| g5 | A10G | 中等训练/推理 |
| p4d/p4de | A100 | 多卡训练 |
| p5/p5e | H100/H200 | 大模型与大规模 RL |

- **计费**：On-Demand / Spot（最高约 90% 折扣）/ Reserved / Savings Plans
- **运维**：需自行管理 AMI、驱动、集群（或用 SageMaker 托管层）
- **定价**：通常高于 RunPod/Lambda 等 neocloud，胜在生态与 SLA

## 对 wiki 的映射

- 实体页：[aws-ec2-gpu.md](../../wiki/entities/aws-ec2-gpu.md)
- 统一选型：[international-gpu-cloud-platforms.md](../../wiki/comparisons/international-gpu-cloud-platforms.md)
