# MACaQ 官方代码：量化与推理入口

- **类型：** repo
- **代码：** https://github.com/huawei-csl/MACaQ
- **论文：** https://arxiv.org/abs/2611.10649v2
- **官方交互介绍：** https://huawei-csl.com/blog/macaq/
- **模型集合：** https://huggingface.co/collections/huawei-csl/macaq
- **入库日期：** 2026-11-16
- **核对范围：** 仓库 README.md、vllm/README.md、vllm/cli.py、quantize/macaq_mxfp4.py（mxfp4_realquant_h、_capture_hooks、streaming_quantize、parse_args）、quantize/macaq_balance.py、vllm/macaq_plugin.py 及目录入口；其他子后端仅依 README 定位。
- **通读：** 已完成
- **沉淀到 wiki：** [MACaQ 论文详情](../../wiki/entities/paper-macaq.md)

## 开源结论

**已开源**：GitHub 仓库为 Apache-2.0，提供可运行的量化与推理 CLI，包含 vLLM、llama.cpp、Diffusers、PyTorch 等分后端集成。**不代表本次已实际执行任何后端或测试所有模型**。Ascend 侧为 CANN 相关 PR 核心算子，README 明示仍需与完整模型推理运行时集成。

## 源码级运行路径：vLLM dense Qwen3

1. 环境按 `vllm/README.md`：Linux、Python 3.12，`pip install ... -e .` 注册 worker plugin。
2. `vllm/cli.py quantize --model Qwen/Qwen3-4B --calibration data/text-calibration.txt --output models/qwen3-macaq --gpu 0` 验证校准文本和全新输出目录，根据 AutoConfig 对模型路由。
3. 对 **dense Qwen3** 子路径，CLI 以子进程调用 `quantize/macaq_mxfp4.py`（`--calib-file`、`--save_dir`、`--activation-precision` 等）；MoE 及 hybrid 走其他脚本，不能一概套用这条路径。
4. `streaming_quantize`：将主体置 CPU、准备校准 hidden，逐 transformer block 搬到 CUDA；`_capture_hooks` 在各 nn.Linear 上注册 forward hook，累积 `H=ΣXᵀX`。
5. `mxfp4_realquant_h`：平衡缩放→块 Hadamard→Hessian 加权指数筛选→`flip_refine_levels`→`pack_mxfp4_from_signed_index`；用 `OCPDenseW4A4Linear` 替换原层，并保存量化 checkpoint。
6. `vllm/cli.py run` 或 `serve`：检查已保存的 `config.json` / quantization_config，注册 `macaq_plugin`；`run` 使用 vLLM `LLM.generate`，`serve` 启动 vLLM OpenAI 兼容 API 服务。**不需每次重做量化**。

## 代表性官方命令

```bash
python3.12 examples/make_smoke_calibration.py --output data/text-calibration.txt
.venv-vllm/bin/python vllm/cli.py quantize \
  --model Qwen/Qwen3-4B --calibration data/text-calibration.txt \
  --output models/qwen3-macaq --gpu 0
.venv-vllm/bin/python vllm/cli.py run \
  --model models/qwen3-macaq --prompt 'Explain quantization.' --gpu 0
```

> 需先按官方 README 配置隔离环境、依赖、模型和设备；以上为入口摘录，不是本次运行记录。量化输入与评测输入应独立准备。
