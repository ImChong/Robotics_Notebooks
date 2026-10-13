# VioLA 项目页

> 来源归档

- **标题：** VioLA — Learning Generalist Humanoid Control Policies from Human Data
- **类型：** site（论文项目页、方法说明与实验演示）
- **URL：** <https://viola.is.tue.mpg.de/>
- **论文：** <https://arxiv.org/abs/2610.12435>
- **代码 / 检查点：** 页面截至 2026-10-13 标注 “Code coming soon”；当前页面没有公开仓库或检查点链接。
- **机构：** Vesoma、ETH Zürich、MPI-IS、University of Tübingen、ELLIS Institute
- **平台：** Unitree G1
- **核查日期：** 2026-10-13
- **一句话说明：** VioLA 以共享 body / hand motion latent 作为策略输出空间，将人类与机器人演示统一成策略监督标签，并通过冻结控制器在 G1 上执行。

## 项目页补充要点

- **真实机器人结果：** 项目页展示 locomotion / posture 30/30（六项任务），whole-body manipulation 31/35（七项主要任务）；每项 5 次，单台 G1。
- **zero-shot 的定义：** 不做任务专属 fine-tuning；任务家族进入过训练，留出的是评测 episode、实验室环境和物体实例。
- **人类数据：** 140.6M 训练帧中 93.2% 来自带动作标注的人类记录；human-only policy 的 locomotion 结果为 12/30（40%）。
- **方法：** VioLA 同时预测 body 和 hand latents；冻结的 body controller / hand decoder 产生关节命令。实时 chunking（RTC）让新 chunk 与推理期间已排队执行的 latent 保持连续。
- **当前开放状态：** 项目页的 Code 按钮仍为 coming soon。论文称代码和检查点将来发布，当前不应将其写成已经开源。

## 对 Wiki 的映射

- [paper-viola.md](../../wiki/entities/paper-viola.md) — 方法、实验、局限及复现状态。
- [viola_arxiv_2610_12435.md](../papers/viola_arxiv_2610_12435.md) — 论文摘录与数据口径。
