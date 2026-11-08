# jepa-wms — 官方代码仓库核验

- **仓库：** <https://github.com/facebookresearch/jepa-wms>
- **核查日期：** 2026-11-08
- **代码范围：** 提供 DINOv3 DROID/RoboCasa 模型、DINOv2 仿真模型，训练/规划配置与 checkpoint；需要按 README 安装专门依赖。
- **可见运行入口：** `python -m app.main --fname ... --debug；configs/vjepa_wm/；evals/；hubconf.py`
- **核对范围：** 阅读官方 README 与目录/部分训练示例；未完整审计代码调用链或实际运行。
- **通读：** 未完成
- **注意：** 运行、模型权重和第三方依赖需按对应仓库的最新 README 验证。
