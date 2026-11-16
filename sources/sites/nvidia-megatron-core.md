# NVIDIA Megatron Core Developer Guide / 英伟达 Megatron Core 官方开发文档

> 官方技术文档来源归档（site）。

- **机构：** 英伟达（NVIDIA）
- **类型：** 官方文档
- **主页：** https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html
- **并行指南：** https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- **第一轮训练：** https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html
- **代码：** https://github.com/NVIDIA/Megatron-LM（已开源）
- **入库日期：** 2026-11-13
- **核对范围：** 官方开发指南导航、并行策略指南与 quickstart（不同模块组合的应用与启用方法）；与 GitHub README/训练入口交叉核实。未读取整个 API reference 的每个子页面，不能以此推断所有算子细节。
- **通读：** 已完成

## 核心分工

- **Megatron-LM：** 集成 Megatron Core 的参考训练项目及预配置入口。
- **Megatron Core：** 供自定义训练框架调用的模型、并行、优化器、checkpoint 等组合组件。
- **TP：** 沿模型层内权重矩阵维度切分；**PP：** 分配网络深度；**DP：** 按训练 batch 复制模型并分数据；**CP：** 按序列维切分上下文；**EP：** 拆分 MoE 专家。
- **使用约束：** TP、EP 等组合下，官方文档特别指出启用 sequence parallel 的要求，实际配置以相应 release 文档和模型示例为准。
- **运行入口：** 按 quickstart 准备 tokenizer/data 与运行环境，参考仓库训练脚本/官方示例启动 torchrun 分布式进程。

## 对 wiki 的映射

- [Megatron-LM 独立论文详情](../../wiki/entities/paper-megatron-lm.md)
- [主仓代码归档](../repos/megatron-lm.md)
- [原始论文归档](../papers/megatron_lm_arxiv_1909_08053.md)
