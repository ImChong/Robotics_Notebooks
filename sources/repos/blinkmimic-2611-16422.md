# BlinkMimic: Training a Humanoid Motion Tracker in Minutes on a Single GPU 官方源码核对

- **类型：** 官方源码仓
- **代码：** https://github.com/fuyichen0725/blinkmimic
- **论文：** https://arxiv.org/abs/2611.16422
- **公开运行入口：** `bash scripts/train.sh logs/run_s0 --seed 0`
- **相关模块：** `blinkmimic.train`
- **运行产物：** 策略 checkpoint、训练日志
- **核对范围：** 官方 README、该仓库公开入口脚本与关键模块的静态核查；外部数据和模型依赖未执行。
- **通读：** 已完成
- **wiki：** [论文独立详情](../../wiki/entities/paper-blinkmimic-2611-16422.md)
- **原文：** [论文来源](../papers/blinkmimic-2611-16422.md)

验证来自仿真 Unitree G1 与固定动作库，没有真实机器人部署证据；耗时受 GPU 型号和评测口径制约。
