# CrossBFM: Distilling a Shared Latent Behavior Space Across Humanoid Embodiments（arXiv:2609.38087）

> 来源归档（ingest · 原文通读）

- **标题：** CrossBFM: Distilling a Shared Latent Behavior Space Across Humanoid Embodiments
- **作者：** Tan-Dzung Do, Tuan Dat Phuong, Nico Bohlinger, Cuc T. Trinh, Siwei Ju, Vien Anh Ngo, Jan Peters, Xinchao Wang, An T. Le（VinRobotics / NUS / TU Darmstadt）
- **arXiv：** <https://arxiv.org/abs/2609.38087>
- **通读：** 已完成
- **核对范围：** 已通读 `arXiv:2609.38087v1` 全文（75,958 字符），核对跨具身统一编码器架构、阶段 1 冻结源 BFM 潜空间蒸馏、阶段 2 latent-conditioned tracker（200 Hz 内部仿真步进 / 50 Hz decimation 4）与 LAFAN / 41 任务奖励优化评测，以及未见具身泛化实验与附录推理管线。

## 核心摘要

- 将行为基础模型（BFM）中通过 Forward-Backward 表示习得的**潜行为空间（latent behavior space）**作为可迁移资产，利用运动重定向提供的帧级跨具身对应关系，在不到 **1 GPU-hour** 内以无机器人专属参数的统一编码器将其蒸馏到新具身，随后接入常规隐变量条件全身跟踪器实现 motion tracking、goal reaching、reward optimization 及 flow-generated latents 四种提示模式。

## 对 wiki 的映射

- [`wiki/entities/paper-crossbfm-shared-latent-behavior.md`](../../wiki/entities/paper-crossbfm-shared-latent-behavior.md)
