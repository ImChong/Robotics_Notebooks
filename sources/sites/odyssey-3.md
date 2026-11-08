# Meet Odyssey-3: Our Most Powerful Foundation World Model

> 来源归档（ingest · Odyssey 官方第二次技术说明；非论文）

- **类型：** official release / project page
- **URL：** <https://odyssey.systems/meet-odyssey-3>
- **发布：** 2026-10-08
- **作者：** Oliver Cameron、Jeff Hawke
- **核查日期：** 2026-11-08
- **核对范围：** 已通读官方文章全部技术正文，从「Generates Environments in Real Time」至「How We Built Odyssey-3」及「Experience Odyssey-3 Today」，并核对 Physics-IQ、WorldMark、机器人、人形、驾驶、多传感器及训练流程段落；未从宣传视频推断未公开的量化指标。
- **通读：** 已完成
- **一句话说明：** 解释 Odyssey-3 的实时交互世界生成、基于世界表示的控制策略适配、三阶段训练与官方自报基准。
- **代码：** 官网未提供可公开下载的 Odyssey-3 训练/推理源码
- **权重：** 官网未提供公开权重下载
- **数据：** 官网未提供完整预训练数据集
- **相关首发文章：** <https://odyssey.systems/introducing-odyssey-3>
- **沉淀到 wiki：** [Odyssey-3](../../wiki/entities/odyssey-3.md)

## 核心原理：预训练—自回归—蒸馏

1. **预训练数据：** 互联网视频与时序定位且经过 schema 校验的事件注释；含对齐键鼠操作的游戏录像；含 caption/metadata 的模拟刚体交互。
2. **生成主干：** 使用多步视频扩散 Transformer，在时间精确的 prompt/control 条件下预测后续视觉状态。
3. **自回归扩展：** 使用 teacher forcing 和 causal masking，从历史观测继续预测后续状态，并由动作输入条件化。
4. **实时推理：** 后训练结合 distribution matching 与 adversarial distillation，蒸馏为少步生成版本。
5. **物理系统适配：** 在设备专属观察—动作样本上训练 action decoder / policy，将世界模型学到的表示映射为设备动作。技术描述不提供实际网络尺寸或完整可复现运行配置。

## 评测（均为官方披露口径）

| 场景 / 指标 | 官方披露值 | 解释边界 |
|---|---|---|
| Physics-IQ Verified V2V | **Odyssey-3 Pro 66.1** | 官方称 2026-10-07 排行榜最高；不是机器人任务成功率 |
| Physics-IQ I2V | **Odyssey-3 Pro 54.7** | 与 V2V 不同任务，不可混写 |
| WorldMark 第一人称风格化 | **Odyssey-3 77.2，排名第一** | 官方取 13 个指标均值 |
| WorldMark 第三人称真实 | **Odyssey-3 79.0，排名第一** | 官方协议 |
| WorldMark 第三人称风格化 | **Odyssey-3 76.3，排名第一** | 官方协议 |
| WorldMark 第一人称真实 | **Odyssey-3 80.6，非第一** | 不可声称四项全胜 |
| 多摄像头生成 | 训练中 checkpoint 在 **100 步**后可同时生成前 3 视角 | 早期视觉实验，不构成传感器精确性标准 |

官网图示的评测条件涉及 video-to-video、image-to-video、模型分辨率与 GPU 成本假设，不能将不同 protocol / 配置直接归为模型绝对优劣。

## 对机器人研究者的含义

- 共享的世界表示可能提升 task-specific policy 的数据效率，但必须独立报告控制成功率、延迟、碰撞、恢复能力与跨本体迁移。
- Flexion 的人形控制包含其自己的机器人学习及全身控制研发；Odyssey-3 本身不是开箱即用的 WBC/关节命令生成器。
- 官网已开放 **research preview** 及开发者接洽入口，但未公开复现所需的训练代码、权重与数据；API 或体验入口不等于开源。

## 一手入口

- [官方发布全文](https://odyssey.systems/meet-odyssey-3)
- [2026-09-15 官方首发](https://odyssey.systems/introducing-odyssey-3)
- [Odyssey 官网](https://odyssey.systems/)
- [在线体验](https://experience.odyssey.systems/)
- [开发者入口](https://developer.odyssey.ml/)
