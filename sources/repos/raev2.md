# RAEv2：作者官方 PyTorch 代码与复现入口

- **仓库：** https://github.com/nanovisionx/RAEv2
- **类型：** repo / PyTorch / RAE / DiT / NWM
- **论文：** https://arxiv.org/abs/2605.18324v2
- **项目：** https://raev2.github.io/
- **数据：** https://huggingface.co/datasets/nanovisionx/RAEv2-data
- **模型：** https://huggingface.co/nyu-visionx/RAEv2-models
- **核查日期：** 2026-11-19
- **核对范围：** 阅读 README（安装、数据、模型、Stage1/Stage2 的 train/eval/权重命令）；阅读 `src/train_stage1.py`、`src/stage1/engine.py`、`src/stage1/rae.py`、`src/train.py`、`src/stage2/engine.py`、`configs/stage2/training/nwm-dinov3l-k7.yaml`。
- **通读：** 已完成

## 开源状态

**已开源，存在明确可运行入口：** README 给出 `uv sync`、`uv run torchrun --nproc_per_node=8` 的分布式训练方式、阶段配置和离线评价 CLI；仓库包含实现源码。已核对代码及命令，**未在当前环境下载权重、训练或复现论文表格**。

## Stage1：冻结视觉编码器，训练重建解码器

- 命令：`uv run torchrun --nproc_per_node=8 src/train_stage1.py --config configs/stage1/training/dinov3l-k7-imagenet.yaml --results-dir ckpts/stage1 --precision bf16`。
- `train_stage1.py` 创建 data loader、RAE、LPIPS 与 discriminator，冻结 encoder、训练 decoder。
- `stage1/engine.py::train_one_epoch` 计算 L1 重建、LPIPS 与可选的 GAN 损失，更新 decoder/判别器以及 EMA；不是训练冻结视觉骨干。
- 训练后经 `scripts/stage1/extract_decoder.py` 提取 EMA decoder，`scripts/stage1/compute_encoder_stats.py` 计算 latent normalization stats，供 Stage2 加载。

## Stage2：flow matching + REPA

- 命令：`uv run torchrun --nproc_per_node=8 src/train.py --config configs/stage2/training/imagenet-dinov3l-k7.yaml --results-dir ckpts/stage2 --precision bf16`。
- `src/train.py` 加载冻结 RAE、REPA target encoder、可选文本编码器、DiT 模型与 EMA；`src/stage2/engine.py` 把 RGB 送入 `rae.encode`，计算 `z_clean`，并以 `transport.training_losses` 获取 flow loss 与 `loss_repa`。
- Navigation world model 配置 `configs/stage2/training/nwm-dinov3l-k7.yaml` 设 `conditioning.type: nwm`，输入 4 帧历史、动作增量，模型输出为 latent future frame；评测 FID/LPIPS。论文另外报告 FVD（不是此 YAML 唯一评价项）。
- 仓库还列出 T2I 训练配置、`src/offline_eval.py`、`src/offline_eval_stage1.py` 与 `scripts/stage1/sample.py` 评测/采样入口。

## 工程边界

- README 的 8 进程示例意味着分布式多卡路径；实际显存/运行时间不应从纸面参数直接推断。
- 数据/权重下载路径需与具体 YAML 的 `data_dir` 与 `pretrained_decoder_path` 对齐；同一论文不同任务需各自选择配置。
- RAEv2 图像生成指标不等于机器人真机策略效果；NWM 的未来图像生成以动作条件驱动，仍未验证实际闭环控制。

## 关联资料

- [论文通读归档](../papers/raev2_improved_baselines_arxiv_2605_18324.md)
- [项目主页归档](../sites/raev2-github-io.md)
- [RAEv2 论文知识节点](../../wiki/entities/paper-raev2-improved-representation-autoencoders.md)
