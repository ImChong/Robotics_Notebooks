# rla-wm — 官方代码仓库核验

- **仓库：** <https://github.com/mlzxy/rla-wm>
- **核查日期：** 2026-11-08
- **代码范围：** 提供数据及权重说明、推理 Notebook、三阶段训练（DINO→图像 UNet；RLA 自编码器；RLA-WM）。
- **可见运行入口：** `train.py --config configs/rla_wm/panda.yaml；docs/training.md；notebooks/inference_demo.ipynb`
- **核对范围：** 阅读官方 README 与目录/部分训练示例；未完整审计代码调用链或实际运行。
- **通读：** 未完成
- **注意：** 运行、模型权重和第三方依赖需按对应仓库的最新 README 验证。
