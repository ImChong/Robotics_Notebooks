# NVIDIA Triton Inference Server：Batchers

- **URL：** <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html>
- **类型：** 官方文档或作者工程文章
- **核验日期：** 2026-11-26
- **通读：** 已完成
- **核对范围：** Dynamic Batcher、配置流程、排队延迟/优先级/超时、Sequence Batcher、Iterative Sequences。

## 来源摘要

Triton 将无状态动态批处理与需要在同一实例处理序列的有状态批处理区分开。动态批处理可设最大排队等待与队列策略，通过实测选择吞吐和延迟取舍。迭代序列需后端支持，不能从服务器提供选项推断所有后端皆可用。

关联：[方法页](../../wiki/methods/ai-model-serving-inference.md)。
