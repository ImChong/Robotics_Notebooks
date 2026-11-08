# dinov3 — 官方代码仓库核验

- **仓库：** <https://github.com/facebookresearch/dinov3>
- **核查日期：** 2026-11-08
- **代码范围：** 官方 README 演示 torch.hub.load 加载 ViT/ConvNeXt 权重，输出 CLS 与 patch token；是视觉特征提取模型，并非带动作输入的世界模型。
- **可见运行入口：** `torch.hub.load(...) / hubconf.py / dinov3/`
- **核对范围：** 阅读官方 README 与目录/部分训练示例；未完整审计代码调用链或实际运行。
- **通读：** 未完成
- **注意：** 运行、模型权重和第三方依赖需按对应仓库的最新 README 验证。
