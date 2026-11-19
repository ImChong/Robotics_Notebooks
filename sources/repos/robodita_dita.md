# Dita（RoboDita/Dita）

> 来源归档（repo）

- **标题：** Dita — Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy
- **类型：** repo / vla / diffusion-transformer / oxe / open-source
- **链接：** <https://github.com/RoboDita/Dita>
- **许可证：** MIT
- **论文：** [arXiv:2503.19757](https://arxiv.org/abs/2503.19757) · 前序 [2410.15959](https://arxiv.org/abs/2410.15959) — [`sources/papers/dita_arxiv_2410_15959.md`](../papers/dita_arxiv_2410_15959.md)
- **项目页：** <https://robodita.github.io/> — [`sources/sites/robodita-github-io.md`](../sites/robodita-github-io.md)
- **入库日期：** 2026-09-28
- **最近复核：** 2026-11-19
- **核对范围：** `README.md`、`scripts/llama_dp.py`、`scripts/vision_tokenizers/vit_tokenizer.py`、`scripts/train_diffusion_oxe.py`、`scripts/train_diffusion_sim.py`、`scripts/finetune_realdata.py`、`scripts/close_loop_eval_diffusion_calvin.py`、`scripts/close_loop_eval_diffusion_libero.py`
- **通读：** 已完成
- **一句话说明：** 基于 PyTorch + Hydra 的 OXE 预训练、CALVIN/LIBERO/ManiSkill2 闭环评测与 Franka 10-shot 真机微调官方仓库。
- **沉淀到 wiki：** [`wiki/entities/paper-dita-scaling-diffusion-transformer-vla.md`](../../wiki/entities/paper-dita-scaling-diffusion-transformer-vla.md)

---

## 开源状态（步骤 2.5，2026-11-19 复核）

| 项 | 状态 |
|----|------|
| OXE 预训练 | `scripts/train_diffusion_oxe.py`（多机 `torchrun` + TensorFlow S3 `openx` 数据流） |
| 仿真训练/评测 | `scripts/train_diffusion_sim.py`（CALVIN `+eval_only=1`、LIBERO、ManiSkill2） |
| 真机微调 | `scripts/finetune_realdata.py`（支持 `use_lora` 开关与 10-shot pkl 数据集） |
| 闭环评测封装 | `scripts/close_loop_eval_diffusion_calvin.py` & `close_loop_eval_diffusion_libero.py`（`PytorchDiffInference`） |
| 数据生成 | `ManiSkill2/openx_utils/` 生成多视角 OXE 格式 ManiSkill2 数据 |
| 权重 | README「Model Checkpoints」提供 Google Drive 下载链（含 OXE/Droid 预训练及 MLP Head 对照） |

## 仓库核心模块（对齐时序图）

| 路径 / 符号 | 角色 | 关键实现细节（源码核对） |
|-------------|------|--------------------------|
| `scripts/llama_dp.py` (`RobotTransformerNet`) | 策略模型核心 | 内含 `RT1ViTImageTokenizer`（DINOv2 + FiLM Q-Former）、`LlamaForCausalLM` 因果主干、`SinusoidalPosEmb`、`DDPMScheduler`/`DDIMScheduler`，可选 `DenoisingMLP`（`AdaLN` 残差块）对比分支 |
| `RobotTransformerNet.forward` & `inference` | 训练与去噪采样 | `forward` 拼接语言、视觉（支持缓存 `reused_context_image_tokens`）、时间步与零填充噪声动作 token；`inference` 按 `DENOISING_STEPS` / `num_inference_steps` 迭代 `noise_scheduler_eval.step(...)`，仅在首步编码图像特征并在后续去噪步复用 |
| `scripts/vision_tokenizers/vit_tokenizer.py` (`RT1ViTImageTokenizer`) | 多视角视觉词元化 | 提取静态图与可选腕部相机特征，通过 `QFormer`（FiLM 文本条件）输出紧凑视觉 token |
| `scripts/close_loop_eval_diffusion_{calvin,libero}.py` (`PytorchDiffInference`) | 闭环推理封装 | 管理滑动观测缓存与 CLIP 文本嵌入（支持 CFG 双路批处理），调用 `model.inference` 并转换 7D 增量/绝对末端位姿 |
| `scripts/train_diffusion_oxe.py` / `train_diffusion_sim.py` / `finetune_realdata.py` | 三大训练入口 | 分别负责 OXE 分布式预训练、仿真微调与真机 few-shot 微调 |

## 关联资料

- [`sources/papers/dita_arxiv_2410_15959.md`](../papers/dita_arxiv_2410_15959.md)
- [`sources/sites/robodita-github-io.md`](../sites/robodita-github-io.md)
