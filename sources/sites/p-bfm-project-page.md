# P-BFM 项目页

> 来源归档（ingest）

- **标题：** P-BFM: Perceptive Behavioral Foundation Models for Humanoid Control via Unsupervised Reinforcement Learning
- **项目页：** <https://roboparty.github.io/P-BFM/>
- **论文：** [arXiv:2610.15833](https://arxiv.org/abs/2610.15833) · [HTML 全文](https://arxiv.org/html/2610.15833v1)
- **代码仓库：** <https://github.com/Roboparty/P-BFM>
- **演示：** 项目页提供浏览器交互仿真、地形适应、遥操作、扰动恢复、目标到达、动作跟踪与奖励推断视频。
- **开源状态核查：** 截至 2026-10-27，`Roboparty/P-BFM` main 分支已含 `pbfm/`、`scripts/`、`tests/` 与 TRAINING/INFERENCE/ARCHITECTURE 文档，支持 G1（29 DoF）与 RP1（24 DoF）训练、tracking/goal/reward 推理和 ONNX 导出（最近提交 2026-10-22）；预训练权重与 RP1 动作 PKL 不随仓库提供。2026-10-15 版本曾标注“代码尚未上传”，已过期。pages 分支存项目网站、论文、图片与演示视频。
- **项目页摘要：** 单一冻结策略在 Unitree G1 上支持多种行为接口；深度感知负责地形适配，显式根指令控制朝向和平面速度。训练使用 40 段平地 LaFAN 序列作为运动提示，地形适配通过在线仿真交互学习，无需地形专属演示。
- **核查日期：** 2026-10-27（初次入库 2026-10-15）

