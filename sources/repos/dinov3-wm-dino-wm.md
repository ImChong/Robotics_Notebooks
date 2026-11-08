# dino-wm — 官方代码仓库核验

- **仓库：** <https://github.com/gaoyuezhou/dino_wm>
- **核查日期：** 2026-11-08
- **代码范围：** 原始 DINO-WM 使用 DINOv2；提供 MuJoCo/PyFlex 数据处理、离线训练和 CEM 规划入口。
- **可见运行入口：** `train.py --config-name train.yaml；plan.py planner=cem；conf/`
- **核对范围：** 阅读官方 README 与目录/部分训练示例；未完整审计代码调用链或实际运行。
- **通读：** 未完成
- **注意：** 运行、模型权重和第三方依赖需按对应仓库的最新 README 验证。
