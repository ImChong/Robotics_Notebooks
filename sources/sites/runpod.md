# RunPod

> 来源归档

- **标题：** RunPod — GPU Cloud for AI
- **类型：** site（国外 GPU 云算力平台）
- **来源：** RunPod Inc.
- **链接：** https://www.runpod.io/ 、https://docs.runpod.io/
- **入库日期：** 2026-07-02
- **一句话说明：** 面向 AI/ML 的 GPU 云：Pods（交互式 GPU 容器）、Serverless GPU 与 Secure/Community 双 tier；按秒计费、200+ 模板、无出站流量费。

## 为什么值得保留

- **机器人/ML 社区高频选择**：卡型覆盖 RTX 4090 至 H100/B200，适合 headless RL 与推理服务。
- **双云 tier**：Secure Cloud（SLA）vs Community Cloud（低价、可中断）对应「生产 vs 实验」。
- **Serverless**：突发推理/批处理无需常驻 Pod。

## 平台要点（文档与公开资料 2026-07 归纳）

| 维度 | 要点 |
|------|------|
| **产品** | GPU Pods、Serverless Endpoints、Instant Clusters（单机多卡） |
| **计费** | 按秒；Community 更便宜 |
| **模板** | 200+ 预构建 Docker 模板（PyTorch、SD、LLM 等） |
| **存储** | Network Volume 持久卷 |
| **网络** | 宣称无 egress 费（以官网为准） |
| **API** | REST + GraphQL |

## 对 wiki 的映射

- 实体页：[runpod.md](../../wiki/entities/runpod.md)
- 统一选型：[international-gpu-cloud-platforms.md](../../wiki/comparisons/international-gpu-cloud-platforms.md)
