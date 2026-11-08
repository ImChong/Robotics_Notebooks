# NVlabs/LongLive：LongLive2.0 官方源码与运行入口

- **类型：** repo
- **仓库：** <https://github.com/NVlabs/LongLive>
- **代码范围：** <https://github.com/NVlabs/LongLive/tree/main/LongLive2.0>
- **使用文档：** <https://github.com/NVlabs/LongLive/blob/main/LongLive2.0/docs/getting_started.md>
- **论文：** <https://arxiv.org/abs/2605.18739>
- **官方项目页：** <https://nvlabs.github.io/LongLive/LongLive2/>
- **许可：** Apache-2.0（查看 LongLive2.0/LICENSE；第三方模型/依赖另计）
- **开源状态：** **已开源**，包含 AR 与 DMD 训练、T2V/I2V、单机及 SP 推理、NVFP4 与 FP8 PTQ 路线。
- **核对范围：** `LongLive2.0/README.md`、`docs/getting_started.md`、`train.py`、`inference.py`、`inference_sp.py`、`trainer/diffusion.py`、`trainer/distillation.py`、`pipeline/causal_diffusion_inference.py`、`pipeline/self_forcing_training.py`、`model/diffusion.py` 及 `configs/nvfp4/` 实际文件结构；静态代码阅读，未在 NVIDIA GPU 上运行。
- **通读：** 已完成
- **核验日期：** 2026-11-08
- **沉淀到 wiki：** [LongLive-2.0 主实体](../../wiki/entities/paper-longlive-2-nvfp4-long-video-generation.md)

## 官方入口与真实调用链

**AR Teacher-Forcing：**
```bash
cd LongLive2.0
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
  --config_path configs/nvfp4/train_ar_nvfp4.yaml \
  --logdir logs/nvfp4_ar --disable-wandb
```

- `train.py::main()` → `normalize_config` → `DiffusionTrainer(config)`（`trainer/diffusion.py::Trainer`）。
- `Trainer.train()` → `dataloader` / `_sync_batch_for_sequence_parallel` → `train_one_step(batch)` → `sp_helper.encode_raw_video_latents`（SP-aware VAE）与 `build_loss_mask` → `model.CausalDiffusion.generator_loss` → optimizer / checkpoint。

**DMD LoRA：**
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
  --config_path configs/nvfp4/train_dmd_nvfp4_step4.yaml \
  --logdir logs/nvfp4_dmd_step4 --disable-wandb
```

- `train.py` 使用 `ScoreDistillationTrainer`（`trainer/distillation.py::Trainer`）；`Trainer.train()` → `fwdbwd_one_step`，critic 每轮、generator 按设定间隔更新；保存独立 LoRA。独立训练并不要求先在该训练进程读取 AR checkpoint。

**推理：**
```bash
python inference.py --config_path configs/nvfp4/inference_nvfp4.yaml
```

- `inference.py` → `CausalDiffusionInferencePipeline(config)` → 加载 generator/LoRA（有量化时先 merge LoRA）→ 按 backend 量化/物化 → `pipeline.inference(noise, text_prompts, initial_latent?)`。
- `pipeline/causal_diffusion_inference.py`：`encode_prompt_blocks` → `_inference_inner` 分 chunk 运行 `self.generator` + `sample_scheduler.step`；clean chunk 回灌更新 KV cache，scene cut 钉住 shot sink → `vae.model.cached_decode` 或 `streaming_vae` 异步路径 → 返回视频。
- 正式 NVFP4 Quick Start 推荐 `utils.inference_utils.setup_nvfp4_pipeline`，不能直接用 BF16 `pipe.to(dtype=...)` 将量化缓冲转换。
- 官方 SP 入口 `inference_sp.py` → `CausalDiffusionInferencePipelineSP`；**当前入口检测 `kv_quant` 为 true 后打印警告并置 false**（代码第 224–226 行），故**不可声称“SP 推理脚本目前已支持 NVFP4 KV-cache 通信加速”**；论文 Appendix D 是论文方案及测试，release 脚本可用性要另行验收。

## 复现边界

- 先按官方 Docs 下载 Wan2.2-TI2V-5B 组件和匹配的 LongLive-2.0 checkpoint，并构建 `fouroversix`、`utils/kernel` CUDA 扩展；NVFP4 配置下 TransformerEngine vs FourOverSix checkpoint 不能混用。
- 论文 45.7 FPS 只计 diffusion，GB200 单卡基准并有不同的异步 VAE 端到端实验；FP8 H100 路线不等价于 NVFP4 GB200 性能。
- 未执行训练、推理、显存或 FPS 复测；性能数字均为论文/官方报告。
- `Long-WAM/` 与 `LongLive2.0/` 是同仓库不同研究范围，勿将视频基础设施当成机器人动作策略。
