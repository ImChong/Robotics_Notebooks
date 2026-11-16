# BlinkMimic: Training a Humanoid Motion Tracker in Minutes on a Single GPU｜BlinkMimic：单卡分钟级人形动作跟踪

- **类型：** 研究原始论文
- **论文：** https://arxiv.org/abs/2611.16422 （v1）
- **HTML：** https://arxiv.org/html/2611.16422v1
- **PDF：** https://arxiv.org/pdf/2611.16422
- **代码：** https://github.com/fuyichen0725/blinkmimic
- **核对范围：** arXiv 原始版本的研究任务、核心方法与算法、主要实验评测、结论及限制；不将第三方摘要作为通读依据。没有在本次执行模型训练或真机复现。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **wiki：** [论文详情](../../wiki/entities/paper-blinkmimic-2611-16422.md)

## 方法与评测核对

- **研究问题：** 围绕 G1 的 40 段 LAFAN1 动作，将拟合速度、尾部失败练习与单迭代 GPU 运行开销分别优化。
- **方法原理：** 可微世界模型的 rollout 上通过 BPTT 更新策略；使用 Muon、参考动作残差与随训练进度衰减的步长；定期以部署条件 probe 定位真实尾部失败，并用 CUDA Graph 捕捉包含物理仿真和重置的整个迭代。
- **原文结果：** RTX 4090、40 段 LAFAN1，达到 return≥0.82 平均 2.94 分钟，连续三个无失败 checkpoint 约 3.38 分钟（三个种子，官方报告）。
- **局限与范围：** 验证来自仿真 Unitree G1 与固定动作库，没有真实机器人部署证据；耗时受 GPU 型号和评测口径制约。
