# OpenHelix-Team/CapVector

- **标题：** CapVector 官方实现仓库
- **类型：** repo / vla / finetuning
- **URL：** <https://github.com/OpenHelix-Team/CapVector>
- **入库日期：** 2026-05-18
- **配套论文：** [CapVector（arXiv:2605.10903）](https://arxiv.org/abs/2605.10903) — 摘录见 [`sources/papers/capvector_arxiv_2605_10903.md`](../papers/capvector_arxiv_2605_10903.md)

## 一句话说明

论文 **CapVector** 公开的代码入口（OpenHelix-Team）：用于复现 **能力向量抽取、\(\theta_{\text{meta}}\) 合并与下游带正交正则的标准 SFT**；训练超参、脚本与数据接口以 README 与 `configs/` 为准。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-capvector-capability-vectors-vla.md`](../../wiki/entities/paper-capvector-capability-vectors-vla.md)


## 2026-11-15 官方代码入口复核（静态检查）

- **核查范围：** 通读仓库根 README、`capvector-pi05/README.md`，核对 `compute_param_diff.py`、`apply_param_diff.py`、`train_regular_loss_pytorch.py`、`serve_policy.py`、`examples/simple_client/main.py` 核心逻辑；**没有执行 GPU 训练或真实机器人**。
- **仓库结构：** 根目录给出 `capvector-oft/` 与 `capvector-pi05/` 两种实现；以下重点是明确可复现入口的 Pi0.5 路径。

| 阶段 | 官方路径与执行关系 |
|---|---|
| 数据转换 | `capvector-pi05/examples/aloha_real/convert_aloha_data_to_lerobot.py`：将 HDF5 数据转换为 LeRobot 格式；README 要求自行采集任务数据。 |
| 基座模型转换 | `capvector-pi05/examples/convert_jax_model_to_pytorch.py`：从 JAX pi05_base checkpoint 转为 PyTorch safetensors。 |
| 抽取向量 | `capvector-pi05/capvector/compute_param_diff.py`：用训练配置加载两份 `model.safetensors`，分别取 state_dict，浮点参数逐键相减，输出带 `state_dict` 的 `diff.pth`；`--only-vlm` 可过滤动作专家相关键。 |
| 合并 | `capvector-pi05/capvector/apply_param_diff.py`：读 base safetensors 和 diff.pth，计算浮点参数 `base + scale * diff` 后导出 `model.safetensors`；要注意 `--strict-keys`/ `--no-strict-keys`。 |
| 数据统计 | `capvector-pi05/scripts/compute_norm_stats.py`：根据训练配置生成训练数据的归一化统计。 |
| 下游训练 | `capvector-pi05/scripts/train_regular_loss_pytorch.py`：读取配置/数据并初始化模型；`prepare_regularization_context` 按 `regularization_vector_path` 加载向量，并基于当前参数与初始化 anchor 计算正则损失；日志输出 action / regularization / total loss，保存 checkpoint。 |
| 服务 | `capvector-pi05/scripts/serve_policy.py`：`create_trained_policy` 装载 checkpoint，经 `WebsocketPolicyServer` 监听，默认 8000 端口。 |
| 示例客户端 | `capvector-pi05/examples/simple_client/main.py`：通过 `WebsocketClientPolicy` 向服务端发送**随机合成观测**，测接口与延迟；**不是完整真机操控闭环**，真机端需用户实现具体观测与执行接口。 |

**资源边界：** README 提供对照 checkpoint 的 Hugging Face 资源及部分合并后的 meta checkpoint；支持代码运行≠本次已独立复现实验。论文 `α=1.1` 是其消融所用默认值，而 `apply_param_diff.py` README 示例 `--scale 1.0`，复现时必须核对配置；不能把默认示例当作论文全部参数。

[官方 Pi0.5 README](https://github.com/OpenHelix-Team/CapVector/blob/main/capvector-pi05/README.md) · [代码目录](https://github.com/OpenHelix-Team/CapVector/tree/main/capvector-pi05)
