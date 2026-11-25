# chorl0229 / T2Mmamba

- **URL（仓库）：** <https://github.com/chorl0229/T2Mmamba>
- **论文：** [arXiv:2602.01352](../papers/t2m_mamba_arxiv_2602_01352.md)
- **权重：** <https://github.com/chorl0229/T2Mmamba/releases/tag/weights-v1>
- **开源状态：** 作者贡献 MIT；第三方依赖、数据集另有许可。公开训练、评测、生成代码及两数据集 EMA 推理权重；官方说明权重严格加载与校验已通过，但 CUDA 生成及论文分数尚未重新验证。
- **核对日期：** 2026-11-25
- **通读：** 已完成
- **核对范围：** README、docs/USAGE.md、docs/ARCHITECTURE.md、docs/SOURCE_AUDIT.md，以及 scripts/generate.py、utils/model_load.py、models/__init__.py、models/unet.py、models/gaussian_diffusion.py、models/mamba_block.py 的推理路径与 models/Key.py、models/DiffCrossattention.py 的关键实现。

## 运行入口

- [安装与数据](https://github.com/chorl0229/T2Mmamba/blob/main/docs/SETUP.md) 与 [使用指南](https://github.com/chorl0229/T2Mmamba/blob/main/docs/USAGE.md)
- `python tools/verify_weights.py` 校验下载；`python -m scripts.generate --opt_path checkpoints/t2m/t2m_condunet1d/opt.txt --text_prompt "A person walks forward and waves." --frames 196 --output_dir outputs/walk`。
- `scripts.generate` → `load_pretrained_model`（严格加载，包含冻结 CLIP）→ `DiffusePipeline.generate` → `generate_batch` → `T2MUnet.forward_with_cfg` 或普通 forward → scheduler.step → `recover_from_ric` 与 NumPy 输出。
- `accelerate launch -m scripts.train` 训练；`python -m scripts.evaluation` 评测，具体选项见官方使用指南。

## 论文与代码差异

官方 [paper-to-code map](https://github.com/chorl0229/T2Mmamba/blob/main/docs/ARCHITECTURE.md) 与源码说明：实际默认密度阈值为均值加标准差，选中帧为 1+归一化分数；周期信号采用 sig_conv 投影；文本用全局 CLIP 表示，不提供逐词 token attention。生成 CLI 默认 dpmsolver、10 步，论文写 UniPC。已发布权重无优化器状态；CUDA 内核与基准复测未完成。固定分段的两两距离仍有二次复杂度。

## 对 wiki 的映射

- [T2M Mamba](../../wiki/entities/paper-t2m-mamba.md)
