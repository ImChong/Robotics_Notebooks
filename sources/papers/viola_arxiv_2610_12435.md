# VioLA: Learning Generalist Humanoid Control Policies from Human Data（arXiv:2610.12435）

> 来源归档（ingest）

- **标题：** VioLA: Learning Generalist Humanoid Control Policies from Human Data
- **类型：** paper / humanoid control / VLA / human demonstrations / motion latents
- **arXiv：** <https://arxiv.org/abs/2610.12435>（cs.RO，v1，2026-10-08）
- **PDF：** <https://arxiv.org/pdf/2610.12435>
- **项目页：** <https://viola.is.tue.mpg.de/>
- **代码/检查点：** 截至 2026-10-13，项目页标注 “Code coming soon”，尚未公开链接；论文称代码与检查点将发布。
- **作者：** Mert Albaba、Jens Beißwenger、Anna Manasyan、Daniel Marta、Michael J. Black、Wieland Brendel、Andreas Krause、Georg Martius、Martin Riedmiller
- **机构：** Vesoma、ETH Zürich、Max Planck Institute for Intelligent Systems（MPI-IS）、University of Tübingen、ELLIS Institute
- **平台：** Unitree G1；手部模块覆盖 Inspire 与 Dex3 解码器
- **入库日期：** 2026-10-13
- **一句话说明：** 用冻结的人体/机器人共用动作编码器，将大规模带标注的人类动作变成与机器人控制器同空间的 body/hand latent 监督，让通用策略预测动作潜变量，再由冻结解码器转成关节命令。

## 核心摘录（面向 Wiki 编译）

### 1. 共享动作空间取代直接关节预测

VioLA 的动作不是关节角，而是拼接后的 128 维动作潜变量：64 维 body latent + 64 维 hand latent。人体与机器人各有配对编码器，将两种来源的身体 / 手部运动映射到对应的共享潜空间；与之配对的机器人 body controller 和 hand decoder 将策略输出解码为关节命令。编码器和解码器预训练后冻结，通用策略训练阶段只更新策略本身。body 侧继承 SONIC 的运动编码器与控制器，hand 模块单独预训练。

**对 Wiki 的映射：** [paper-viola.md](../../wiki/entities/paper-viola.md)「核心机制」；[SONIC](../../wiki/methods/sonic-motion-tracking.md) 的潜变量控制接口。

### 2. 人类视频如何成为机器人策略标签

一段带图像、指令、人体 / 手部动作标注的人类演示，经冻结编码器生成 body 与 hand latent 标签；机器人演示也以机器人编码器进入同一目标空间。人类演示没有机器人本体感知状态，因此研究者用冻结 body controller 在仿真中跟踪该人体动作，生成与人类图像时间对齐的机器人 proprioception，并按机器人当前朝向构造 body target。随后，原始人类图像、语言指令、对齐的机器人状态和动作 latent chunk 共同监督策略。没有 hand annotation 的样本只监督 body 输出。

训练池共 31,988 个 episode、140,572,312 帧（约 780.96 小时），其中 93.2% 为人类记录。作者估算其训练 schedule 下人类数据约占训练样本 exposure 的 80%；“93.2% 帧数”不等于每个训练阶段都按此比例采样。

**对 Wiki 的映射：** [paper-viola.md](../../wiki/entities/paper-viola.md)「训练数据与标签构造」；[human-data supervision 概念关联](../../wiki/concepts/imitation-learning.md)。

### 3. 真机表现与证据边界

论文在单台 Unitree G1 上评测 13 个主要任务，每项 5 次。无需任务专属微调时，locomotion / posture 为 30/30（100%）；whole-body manipulation 为 31/35（88.6%）。另加 bottle-carrying 后，该扩展任务 0/5，合并 manipulation 结果为 31/40（77.5%）。同配方在 GR00T N1.7、π0.5 和 DiT4DiT 三类 backbone 上测试；GR00T 版本在主要结果中表现最好。human-only 策略在真实机器人 locomotion 为 12/30（40%）。人类先训练再混合训练有利于 locomotion（80%→100%），但 manipulation 上 robot-only 为 94.3%，高于 human-then-mixed 的 88.6%。

论文所称 **zero-shot** 指评测任务无需任务专属微调；任务家族仍出现在训练中，评测 episode、实验室环境与物体实例被留出。当前实验仅覆盖一台 G1 与有限试次，不能外推成跨机器人、任意新任务的零样本泛化结论。

**对 Wiki 的映射：** [paper-viola.md](../../wiki/entities/paper-viola.md)「实验结果」「结论」与「局限与风险」。

## 归档说明

本条仅收录论文与作者项目页公开信息。项目页显示代码 coming soon；在公开代码 / 检查点链接出现前，不将其标注为已开源或可复现代码。
