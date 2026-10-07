# nv-tlabs/PixelUMM

- **名称：** PixelUMM 官方实现
- **类型：** repo / unified-multimodal-model / pixel-space
- **URL：** <https://github.com/nv-tlabs/PixelUMM>
- **项目页：** [PixelUMM](../sites/pixelumm-project.md)
- **配套论文：** [PixelUMM（arXiv:2609.38597）](../papers/pixelumm_arxiv_2609_38597.md)
- **模型权重：** <https://huggingface.co/nvidia/PixelUMM>
- **项目实体：** [PixelUMM](../../wiki/entities/paper-pixelumm.md)
- **入库日期：** 2026-10-07
- **开放状态：** **代码已公开，权重另行提供且为非商业许可。** 官方仓库有推理入口、数据处理、评测工具及四步 toy training 示例。模型权重需从 Hugging Face 单独下载，使用 NVIDIA One-Way Noncommercial License。源码以 Apache-2.0 为主，但个别源文件保留不同第三方许可声明。

## 仓库提供什么

- `inference.py` / `inference_batch.py`：图像和视频理解、文本到图像及文本到视频入口。
- `modeling/`：Qwen3 主干、PixelUMM MoT 路由、像素生成与采样组件。
- `data/`：图像/视频解码、尺寸处理和数据配置。
- `eval/`：图像/视频理解评测工具。
- `train_toy.py` 与 `TRAIN.md`：四任务 toy training。官方明确不是完整训练配方；示例估算需约 7 张至少 48 GiB GPU，产物约 61 GB。

## 复现注意

1. 按 `ENVIRONMENT.md` 建立 Linux、NVIDIA GPU、CUDA/PyTorch 与 FlashAttention 环境。
2. 按 `CHECKPOINT.md` 分别准备 PixelUMM 权重和 Qwen3-8B config/tokenizer 文件；官方说明 PixelUMM checkpoint 已含学习到的语言模型权重，不需额外 Qwen 权重分片。
3. 先执行 `check_checkpoint.py` 检查 checkpoint 完整性与 tensor 兼容，再按 `inference.py` 任务入口推理。
4. 单条 T2V 推理默认使用 Cosmos guardrail；该模型需单独获准访问。README 说明批量入口不运行该 guardrail，使用时需自行落实安全审核。
5. Toy training 数据需另行下载与校验；不可把该示例等同于论文完整训练流程。

模型许可与源码许可不同，不应把 checkpoint 的非商业限制误读成仓库代码许可，也不能仅凭 Apache-2.0 代码许可推断权重可商用。
