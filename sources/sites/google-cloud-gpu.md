# Google Cloud GPU

> 来源归档

- **标题：** Google Cloud GPU / Vertex AI
- **类型：** site（超大规模云 GPU 算力）
- **来源：** Google Cloud
- **链接：** https://cloud.google.com/compute/docs/gpus 、https://cloud.google.com/vertex-ai
- **入库日期：** 2026-07-02
- **一句话说明：** GCP 计算引擎 GPU VM 与 Vertex AI 托管 ML：NVIDIA L4/A100/H100 与 **TPU v5e/v5p/v6e** 并存；适合 GCP 原生团队、JAX/TensorFlow TPU 训练与 Vertex 端到端 MLOps。

## 为什么值得保留

- **本库历史入口**：legacy 资源地图列 Google Cloud。
- **唯一主流 TPU 出租**：Transformer/JAX 大规模训练备选路径。
- **与 Colab 协同**：Colab 运行时即 Google 基础设施延伸。

## 平台要点（公开资料 2026-07 归纳）

| 维度 | 要点 |
|------|------|
| **GPU VM** | a2（A100）、a3（H100）、G2（L4）等 |
| **TPU** | v5e/v5p/v6e；Vertex AI TPU 节点 |
| **Vertex AI** | 托管训练/推理/流水线；L4 推理价常低于 AWS G5 |
| **计费** | 按秒；Preemptible/Committed Use Discount |
| **选型** | 数据已在 GCS、要用 TPU 或 Vertex 时优先 |

## 对 wiki 的映射

- 实体页：[google-cloud-gpu.md](../../wiki/entities/google-cloud-gpu.md)
- 统一选型：[international-gpu-cloud-platforms.md](../../wiki/comparisons/international-gpu-cloud-platforms.md)
