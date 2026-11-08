# flowwm — 官方代码仓库核验

- **仓库：** <https://github.com/facebookresearch/Flow-World-Models>
- **核查日期：** 2026-11-08
- **代码范围：** 提供 Waymo 和合成数据集训练配置；使用 PyTorch distributed torchrun 启动 flow matching 世界模型，仓库许可证 CC BY-NC 4.0。
- **可见运行入口：** `main.py / configs/dinov3/flow_matching_rae_waymo/... / trainer/ / models/`
- **核对范围：** 阅读官方 README 与目录/部分训练示例；未完整审计代码调用链或实际运行。
- **通读：** 未完成
- **注意：** 运行、模型权重和第三方依赖需按对应仓库的最新 README 验证。
