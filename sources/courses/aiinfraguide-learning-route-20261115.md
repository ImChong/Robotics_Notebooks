# AIInfraGuide 官方原文：AI Infra 学习路线

- **类型：** course / 官方完整学习路线
- **一手文档：** <https://github.com/caomaolufei/AIInfraGuide/blob/main/docs/guides/AI%20Infra%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF.md>
- **所属官方项目：** <https://github.com/caomaolufei/AIInfraGuide>
- **校对版本：** GitHub blob `dc852a2d2142abb150e3f20941453185cf59834e`；页面 frontmatter `updatedDate: 2026-11-01`
- **核对日期：** 2026-11-15
- **核对范围：** 已通读上述**单篇官方《AI Infra 学习路线》原文**，从「全景概览」到第零层基础、第一层 CUDA 算子优化、第二层分布式训练、第三层推理部署、性能调优决策树、新人实践路径、参考资料列表。**此处通读完成仅指上述完整学习路线文档；不表示已精读 AIInfraGuide 仓库中的全部分章教材、代码或完成 GPU 实测**。
- **通读：** 已完成

## 关键结构（原文综合）

| 官方层次 | 核心主题 | 作者给出的可验证练习 |
|---|---|---|
| 第零层 | Linux/C++/Python、PyTorch、Transformer、GPU 与集群通信 | 手写训练循环、画出 Transformer、手算参数与通信量 |
| 第一层 | GPU 存储层次、CUDA Warp/Block、Reduce、GEMM、Softmax、FlashAttention、Triton | 从 baseline 到分块和融合；用 Nsight 对比实际带宽、延迟与正确性 |
| 第二层 | DDP、NCCL、ZeRO/FSDP、TP/PP、MoE、上下文并行 | 计算显存/通信账本；画多卡拓扑、测试扩展效率 |
| 第三层 | Prefill/Decode、KV Cache、PagedAttention、连续批处理、量化、推测解码、PD 分离与生产部署 | 测量 TTFT/TPOT、QPS、尾延迟、显存与 Goodput；使用同一工作负载做 A/B |

**系统思维：** 优化须说明资源交换，典型为 ZeRO 通信换显存、激活重算计算换显存、低精度数值误差换带宽、PD 分离的架构复杂度换服务延迟；报告需保留硬件、模型、输入规模、精度和统计口径。

## 具身与机器人部署的迁移边界

- **可以借用**：显存预算、Kernel profiling、多卡同步和端侧量化流程。
- **不应直接迁移**：LLM `tokens/s`、`TTFT` 并非机器人策略输出的闭环 deadline；必须额外测控制频率、尾延迟、通信和安全性。
- **没有独立实测**：原文中的性能练习目标和示例数值是作者提供的教学检验标准，本次未在 GPU 上复现。

## 对 wiki 的映射

- [AIInfraGuide 学习路线与项目知识节点](../../wiki/entities/aiinfraguide.md)。
- [模型架构纵深路线 Stage 5](../../roadmap/depth-model-architecture.md)。
- [LLM 推理显存估算](../../wiki/concepts/llm-inference-vram-estimation.md)。

## 其他阅读状态

官方仓库 README、GEMM 章节概览和 Astro 站点入口核对参见 [仓库来源归档](../repos/aiinfraguide.md)，其余分章教材和潜在运行实验仍需单独精读，原仓库归档继续标注「通读：未完成」。
