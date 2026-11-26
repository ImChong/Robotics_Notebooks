# ONNX Runtime：Quantize ONNX Models

- **URL：** <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>
- **类型：** 官方文档或作者工程文章
- **核验日期：** 2026-11-26
- **通读：** 已完成
- **核对范围：** Quantization Overview、ONNX quantization representation、预处理、动态与静态量化、Quantization Debugging、Method selection。

## 来源摘要

ONNX Runtime 将 8 位线性量化区分为张量级 QDQ 与算子级表示。静态方法用代表数据离线定标，动态方法运行时估计激活参数；文档建议在预处理而非量化阶段组合图优化，以更易比较量化前后对应张量。本文不假定每个执行后端都有同样的低精度算子。

关联：[方法页](../../wiki/methods/ai-model-compression.md)。
