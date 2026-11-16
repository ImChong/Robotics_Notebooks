# TeamFasterwam/FasterWAM — 官方代码

- **代码：** <https://github.com/TeamFasterwam/FasterWAM>
- **项目：** <https://teamfasterwam.github.io/Faster-WAM/>
- **论文：** <https://arxiv.org/abs/2608.02365>
- **状态：** 已发布训练与 benchmark 评测代码；许可证 PolyForm Noncommercial 1.0.0。官网模型权重标记待发布。
- **环境：** README 要求 Linux、CUDA 12.8、PyTorch 2.7.1、BF16 GPU；论文训练配方使用 H200。
- **运行入口：** `bash scripts/train.sh libero 8 ...` 调 `accelerate launch scripts/train.py`；`scripts/train.py` 经 Hydra 调 `fasterwam.runtime.run_training(cfg)`；`bash scripts/eval.sh libero ckpt=...` 转 `experiments.libero.run_libero_manager_v2`。
- **复现要求：** 用 README_Libero.md 准备数据和 text cache、保存 checkpoint 配套 `dataset_stats.json`；不能用两步 smoke run 冒充论文结果。
- **对应 wiki：** [Faster-WAM](../../wiki/entities/paper-rcl-2608-02365-faster-wam-do-world-action-models-need-deep-acti.md)
