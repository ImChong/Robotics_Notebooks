# NVIDIA/Megatron-LM / 英伟达 Megatron-LM

> 官方 GitHub 开源仓库来源归档（repo），2026-11-13 核对。

- **类型：** repo
- **机构：** 英伟达（NVIDIA）
- **代码：** https://github.com/NVIDIA/Megatron-LM（已开源）
- **许可：** Apache License 2.0（仓库 LICENSE；使用前仍应核查各子组件许可）
- **官方文档：** https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html
- **起步教程：** https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html
- **原始论文：** https://arxiv.org/abs/1909.08053
- **入库日期：** 2026-11-13
- **核对范围：** 阅读主仓 README、仓库结构、pretrain_gpt.py、model_provider.py 及 megatron/training/training.py 关键导入与训练入口；结合官方 quickstart 与 parallelism guide 检查运行方法。没有在 GPU 集群上执行训练或验证性能。
- **通读：** 已完成
- **一句话说明：** 当前仓库同时承载 Megatron-LM 参考训练脚本与模块化 Megatron Core，面向 Transformer 的高性能分布式训练。

## 源码身份与入口

| 路径 | 已核对作用 |
|---|---|
| `pretrain_gpt.py` | GPT 训练主入口；解析 args、构造配置，调用 `pretrain(..., train_valid_test_datasets_provider, ..., forward_step)` |
| `model_provider.py` | `model_provider` 使用 builder 构建 `GPTModel` / `HybridModel` |
| `gpt_builders.py` | GPT model builder 路径（主入口导入） |
| `megatron/training/training.py` | 分布式训练工具、优化器、pipeline forward/backward、checkpoint 等训练核心 |
| `megatron/training/datasets/providers.py` | `build_gpt_datasets`，被 `train_valid_test_datasets_provider` 调用 |
| `megatron/core/` | TP/PP/CP/分布式训练、MoE、模型、优化器等核心库 |

`pretrain_gpt.py` 的 `__main__` 先调用 `parse_and_validate_args`，构造 `PretrainConfigContainer` 并调用 `set_run_config`，随后将 dataset provider、model type 和 forward_step 传入 `pretrain`。`forward_step` 获取 batch、前向计算并返回 loss 函数闭包；训练调度/优化器与 checkpoint 由 `megatron/training/training.py` 集中管理。

## 运行状态与可复现边界

- **开源结论：** 仓库代码公开可读取，且存在可辨识的训练入口；非单纯权重或 README 占位仓。
- **推荐复现：** 官方 quickstart 先安装/准备数据、配置分布式启动参数，再按当前版本示例运行；具体 torchrun 参数须匹配可用 GPU 数量、词表与数据路径。
- **本次未执行：** CUDA/NCCL 多机训练、GPU 性能基准与 checkpoint 恢复；因此不宣称复现了 2019 年 512 卡实验。
- **版本意识：** 本归档描述 2026-11-13 公开主干代码结构；原论文 v4 说明 2019 年的实现思想，两者不能视为完全相同版本。

## 对 wiki 的映射

- [Megatron-LM：原论文、框架及现代并行训练](../../wiki/entities/paper-megatron-lm.md)
- [论文归档](../papers/megatron_lm_arxiv_1909_08053.md)
- [开发文档归档](../sites/nvidia-megatron-core.md)
