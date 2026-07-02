# Lambda Cloud（Lambda Labs）

> 来源归档

- **标题：** Lambda Cloud — GPU Cloud for AI
- **类型：** site（国外 GPU 云算力平台）
- **来源：** Lambda Labs, Inc.
- **链接：** https://lambda.ai/service/gpu-cloud 、https://docs.lambda.ai/
- **入库日期：** 2026-07-02
- **一句话说明：** 面向 AI 研究与生产的 GPU 云：预装 Lambda Stack（CUDA/PyTorch/TF），支持 A100/H100 及 **1-Click Clusters**（16–2000+ GPU、InfiniBand），适合多节点分布式训练。

## 为什么值得保留

- **多节点训练标杆**： indie 团队租 8×H100 SXM + NVLink/IB 的常见选择。
- **Lambda Stack**：开箱即用 DL 环境，减少环境拼装。
- **与 Isaac/机器人**：大规模策略预训练、VLA 微调的长跑场景。

## 平台要点（公开资料 2026-07 归纳）

| 维度 | 要点 |
|------|------|
| **卡型** | A100、H100、H200、B200 等；偏高端 |
| **集群** | 1-Click Clusters，16+ GPU 多机 |
| **计费** | 按小时/分钟；预留实例更便宜 |
| **网络** | 部分实例 InfiniBand；无 egress 费（官网宣传） |
| **局限** | 高峰季 H100 库存紧张；即时多卡可能需 RunPod 备选 |

## 对 wiki 的映射

- 实体页：[lambda-cloud.md](../../wiki/entities/lambda-cloud.md)
- 统一选型：[international-gpu-cloud-platforms.md](../../wiki/comparisons/international-gpu-cloud-platforms.md)
