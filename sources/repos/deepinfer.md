# DeepSeek-AI/DeepInfer

> 来源归档（repo）

- **标题：** DeepInfer
- **代码：** <https://github.com/deepseek-ai/DeepInfer>
- **类型：** 大规模模型离线推理框架
- **默认分支：** `main`
- **源码状态：** 已公开
- **许可证：** Apache-2.0
- **核验日期：** 2026-10-20

## 一句话摘要

DeepInfer 是 DeepSeek 面向大规模生产服务的高性能推理框架开源版本，当前仓库重点提供离线推理能力及服务级运行时组件。

## 核心能力

README 列出的能力包括优化算子与模型/采样运行时、Attention DP 与 balanced CP、MoE EP/EPLB、DSpark 与置信度调度验证、SWA 有界重放、异步批调度、batch-invariant 推理、确定性采样和结构化输出。

当前 README 列出的支持模型为 DeepSeek-V4.1-Flash。硬件要求为 NVIDIA SM100 GPU 或 Huawei Ascend 950 NPU；NVIDIA 路径要求 CUDA 13，Ascend 路径要求 CANN 9.2。项目给出的离线示例通过 `examples/offline_inference.py` 启动。

## 对机器人系统的参考价值

该仓库可作为大模型推理运行时、批调度、并行策略与结构化输出的工程参考。README 所列硬件为 SM100 GPU 与 Ascend 950 NPU，且发布内容侧重离线推理；它不是面向机器人边缘计算平台的即插即用部署方案。

## 复现与边界

项目 README 指出，完整推理服务尚未包含 KV cache 系统、VE/Prefill/Decode 解耦以及跨数据并行副本的负载均衡。因此，评估服务化能力时应把这些缺失组件与已实现的引擎功能区分开。构建要求 Linux、GCC 13+、Python 3.12，以及对应加速器的软件栈。

## 公开资源

| 资源 | URL |
|------|-----|
| GitHub 仓库 | <https://github.com/deepseek-ai/DeepInfer> |
| README | <https://github.com/deepseek-ai/DeepInfer/blob/main/README.md> |

## 参考来源

- [DeepInfer 官方 README](https://github.com/deepseek-ai/DeepInfer/blob/main/README.md)
- [GitHub 仓库元数据](https://api.github.com/repos/deepseek-ai/DeepInfer)
