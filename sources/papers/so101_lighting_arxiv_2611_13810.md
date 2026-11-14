# Camera equalisation of an 11.5x drop in environmental illuminance: four imitation-learning policies on an SO-101 arm｜自动曝光如何影响 SO-101 光照鲁棒性评测

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.13810
- **PDF：** https://arxiv.org/pdf/2611.13810
- **核对范围：** arXiv HTML v1 全文：Methods、Results、Discussion、Conclusions 与附录逐回合统计表、图像亮度分析；核对作者发布的数据集入口，未运行分析脚本。
- **通读：** 已完成
- **收录日期：** 2026-11-14
- **代码或数据：** https://huggingface.co/datasets/fysical/so101-lighting-autoexposure

## 核心摘录

1. 四种策略 ACT、SmolVLA、Diffusion Policy 和 π0.5 均用同一抓放任务 50 条演示训练。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-so101-lighting.md)
2. 在约 520 与 45 lux 各运行每策略 10 次。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-so101-lighting.md)
3. 同步记录环境 lux、曝光/白平衡与输入图像统计，避免误用照度当图像域偏移量。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-so101-lighting.md)

## 作者报告的结果与限制

常亮 20/40 成功，低照度 19/40（-2.5 个百分点，95% CI -23 至 +19）；前相机图像均值亮度仅下降约 3%，腕相机约 19%，相机开启自动曝光。

样本量小且曝光未锁定，不能证明策略对真实低照图像干扰鲁棒。

## 原文核对摘录（2026-11-14）

论文使用单个 SO-101 桌面抓放任务、前视与腕视两台相机；ACT、SmolVLA、Diffusion Policy、π0.5 四种策略各用 50 条演示训练，在亮（约 520 lux）/暗（约 45 lux）两个条件分别 10 次试验，共 80 rollout，每种光照共 40 次。亮组 20/40、暗组 19/40，-2.5 个百分点，95% CI 约 [-23,+19] pp，不能判定效果等价。相机自动曝光/自动白平衡启用，前视图像平均亮度只降约 3%、腕视约 19%，显著小于环境照度约 11.5 倍下降。结论应是必须同时记录 lux、相机曝光及模型实际输入图像，而非“机器人策略本质抗低照”。原实验仅单任务与小样本。

## 复现边界

尚未独立运行训练、推理或机器人试验；论文数字为作者报告。对 wiki 的映射：[论文详情](../../wiki/entities/paper-so101-lighting.md)。
