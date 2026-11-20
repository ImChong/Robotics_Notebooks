# GeWu-Lab/Gestalt 源码核读

- **仓库：** https://github.com/GeWu-Lab/Gestalt
- **官网：** https://gewu-lab.github.io/Gestalt/
- **论文：** https://arxiv.org/abs/2610.00576
- **模型：** https://huggingface.co/GeWu-Lab/Gestalt
- **核对日期：** 2026-11-20
- **核对范围：** 阅读 README 的 Quick Start/Data Format，并打开 scripts/training/gestalt_sft.sh、train_stage3.py、scripts/inference/run_mmu.py、run_t2i.py、gestalt/data/stage3.py、gestalt/model/builder.py；核对调用与 I/O，未实际执行 GPU 任务。
- **通读：** 已完成

## 可运行源码入口

| 入口 | 真实调用和边界 |
|---|---|
| scripts/training/gestalt_sft.sh | 校验 DATA_DIR、PRETRAINED、config 和 DeepSpeed，torchrun scripts/training/train_stage3.py |
| scripts/training/train_stage3.py | 使用 Stage3Dataset、TokenBudgetPackingDataset、create_stage3_processor、load_tokenizer、create_trainer |
| scripts/inference/run_mmu.py | 读取原始 VQ-VAE 图像码本 .npy；GestaltModelLM.from_pretrained → 离散扩散生成文字 |
| scripts/inference/run_t2i.py | 加载模型 → generate_t2i → 保存 output_tokens.npy（VQ 码本 ID，并非图片 PNG） |

## 限制

- **源码与权重已开源**，但 Stage-III SFT 需要额外提供满足 schema 的 Parquet 数据；SFT 数据未完整公开。
- 图像问答入口不是原始 jpg/png 即可直接喂入，官方脚本要求图像 token。
- 未在当前环境独立验证 CUDA/显存/生成质量；代码也没有机器人动作闭环控制路径。

## 对 wiki 的映射

[论文独立节点](../../wiki/entities/paper-gestalt-2610-00576.md)
