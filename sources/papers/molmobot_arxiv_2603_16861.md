# MolmoB0T：大规模仿真实现零样本机器人操作 / Large-Scale Simulation Enables Zero-Shot Manipulation

- **类型：** 论文 / 仿真生成数据 / 视觉语言动作策略 / Sim2Real
- **原文：** https://arxiv.org/abs/2603.16861
- **HTML 全文：** https://arxiv.org/html/2603.16861v3
- **版本：** arXiv:2603.16861v3（2026-11-08）；初稿 2026-03-17；官方项目标注 CoRL 2026
- **作者：** Abhay Deshpande、Maya Guru、Rose Hendrix、Snehal Jauhri 等；完整列表见论文原文
- **机构：** Allen Institute for AI（Ai2）、University of Washington、UCLA、TU Darmstadt、Princeton University
- **项目主页：** https://allenai.github.io/MolmoBot/
- **官方博客：** https://allenai.org/blog/molmobot-robot-manipulation
- **模型代码：** https://github.com/allenai/MolmoBot
- **仿真/数据生成代码：** https://github.com/allenai/molmospaces
- **数据：** https://huggingface.co/datasets/allenai/molmobot-data
- **模型权重：** https://huggingface.co/collections/allenai/molmobot-models
- **入库日期：** 2026-11-17
- **核对范围：** arXiv v3 HTML 全文 §§1–6，附录 A（传感器、任务、专家演示、随机化、数据统计）、B（模型、数据混合与增强、平台 IO）、C（真实及仿真评测、数据与模型消融）；逐条对照项目页、Ai2 博客、MolmoBot/MolmoSpaces 官方 README 与部署/训练入口；Molmo2 预训练数据边界另参考 arXiv:2601.10611v4 §2–3、附录 B/F。
- **通读：** 已完成
- **一句话说明：** MolmoBot-Engine 在 MolmoSpaces 中程序化生成约 170 万条专家轨迹，训练 Molmo2-4B + DiT 流匹配动作头等策略，未经真实机器人轨迹训练或任务微调便迁移至 Franka FR3 与 RB-Y1；但 Molmo2 视觉语言基座本身来自真实图像/视频在内的多模态预训练，不能称整个系统从零到尾均纯仿真训练。

## 原文方法摘录

1. **仿真专家数据（§3 / 附录 A）：** 在 MolmoSpaces 场景中采样操作物体与任务，随机化灯光、纹理、动力学、机器人初始位姿、相机内外参；Franka 用 IK 插值，RB-Y1 用 CuRobo 规划；支持失败重试并过滤无效轨迹。
2. **任务与规模（附录 A.7，Table 3/4）：** 约 170 万条专家 demonstrations、5,704 小时、94.2k 独立场景，2 类机器人；官方数据集定义八种配置（平台 × 任务），而 §3.2 方法描述的是 Pick、PnP Into、PnP Next-To、PnP Color、Open、Open-door 六种任务语义，计数口径不可混写。
3. **MolmoBot 结构（§4.1）：** 用经过图文/多图像/视频预训练的 Molmo2-4B 初始化 VLM，冻结 SigLIP 视觉编码器；把语言、当前图像及至多两帧历史、关节本体状态送入逐层 cross-attention 的 DiT flow matching action head；预测 16 步、执行 8 步再重规划。
4. **另外两类策略（§4.2–4.3）：** SPOC 用 SigLIP2 / SigLIP 特征及轻量 Transformer 并行离散动作块；MolmoBot-Pi0 采用 PaliGemma 3B + flow matching action expert，以隔离仿真数据与架构效应。两者均不可误写为从零开始训练的视觉模型。
5. **评测范围（§5）：** DROID/Franka 真机 4 环境 × 10 任务、每任务 3 次，共 120 trials/策略；MolmoBot (H=1) 79.2%，对照 π₀.₅-DROID 39.2%。不同设备/基线训练源不一，结论限定作者协议。
6. **移动操作（§5.1.2 / 附录 C.2）：** RB-Y1 有真机开门、移动抓取与柜门操作演示，但没有适用的公开通用基线，不能照搬桌面 79.2% 作为移动平台的定量成功率。
7. **消融与边界（§5.3、§6）：** 增加 pick 数据量改善真实评测；扩增环境数量效果不单调，作者提出强视觉预训练可能降低新增背景多样性的边际收益。柔性物体、高度接触操作、稳健长期部署超出主要验证范围；论文承认未来可能仍需要目标域真实数据。
8. **“纯仿真”准确性：** §4.1 明写 Molmo2-4B 为大型 image-text 预训练 VLM，且接受多图/视频训练；Molmo2 原论文（https://arxiv.org/html/2601.10611v4）描述真实视频采集与人工标注的数据。因此该研究仅指**机器人策略训练使用的操作轨迹是仿真生成，且没有真机训练/微调**，不指所有初始化权重均未见真实数据。

## 对 wiki 的映射

- [MolmoB0T 独立论文详情](../../wiki/entities/paper-molmobot.md)：原理、对比实验、预训练边界、工程复现。
- [MolmoBot-Data 原有数据集详情](../../wiki/entities/dataset-molmobot-data.md)：将旧二手综述描述的规模纠正至 v3 官方数据统计。
- [Sim2Real 纵深路线 Stage 3](../../roadmap/depth-sim2real.md)：视觉预训练 + 合成轨迹的训练/迁移分工。
