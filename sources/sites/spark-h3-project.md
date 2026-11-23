# Spark-H3：官方项目主页与 Hugging Face 发布

> 一手项目主页来源归档（site）

- **标题：** Spark-H3 — Adaptive Block Sparse Attention for MiniMax-H3
- **类型：** site / 官方技术演示
- **项目主页：** https://zechengtang.github.io/Spark-H3/
- **论文：** https://arxiv.org/abs/2611.24256
- **代码：** https://github.com/zechengtang/Spark-H3
- **Hugging Face：** https://huggingface.co/Aazeus/Spark-H3
- **入库日期：** 2026-11-23
- **核对范围：** 阅读官方项目页技术模块、Reblock/Reweight、Benchmark、LoRA 与 Ref2VA 演示说明；对照 Hugging Face 模型卡 Overview、Benchmarks、Quick Start、ComfyUI、License、Base Model 开放说明。视频演示未逐帧播放，未进行 GPU 复测。
- **通读：** 已完成
- **开源结论：** **已开源（推理实现）**；代码、注意力算子和 ComfyUI 集成公开。Hugging Face 发布的是 Spark-H3 加速后端及使用材料，**不重分发 MiniMax-H3 基础模型权重**；另从上游 MiniMaxAI/MiniMax-H3 下载。研究论文实验与项目页后续性能测试的设备/配置不同，数字不可混用。
- **许可证：** Spark-H3 原创代码 Apache-2.0；MiniMax-H3 上游权重适用其自身 Community License，第三方算子依 NOTICE。

## 官方资料摘录

1. **Reblock：** 对注意力偏好相近的 token 重新分配固定大小 tile，在相同精确块预算下覆盖更高的 attention mass。
2. **Reweight：** 用全局平均 query 校准未选块的加权 KV 代表及 log-mass bias，与 exact 分支同一归一化。
3. **整合：** 提供 Diffusers 上下文管理器、ComfyUI 插件和 Ref2VA 支持，并可叠加社区少步蒸馏 LoRA。
4. **发布边界：** 框架不改变 H3 主干权重；论文 Table 1 的 DiT 2.49× 不等于项目页 ComfyUI 后续独立配置的速度。
5. **运行条件：** 公开 README 以 CUDA/PyTorch 匹配为前提，具体 GPU/ComfyUI 版本随 release 更新；部署前以 GitHub 最新 release/README 为准。

## 对 wiki 的映射

- [Spark-H3 独立论文实体](../../wiki/entities/paper-spark-h3.md)
- [官方源码核对](../repos/zechengtang-spark-h3.md)
- [论文 v1 来源归档](../papers/spark_h3_arxiv_2611_24256.md)
