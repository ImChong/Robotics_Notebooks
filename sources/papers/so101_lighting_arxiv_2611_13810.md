# Camera equalisation of an 11.5x drop in environmental illuminance: four imitation-learning policies on an SO-101 arm｜自动曝光如何影响 SO-101 光照鲁棒性评测

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.13810
- **PDF：** https://arxiv.org/pdf/2611.13810
- **核对范围：** 核对论文 arXiv v1 摘要、用户 2026-11-12 资讯附件、公开数据入口；未通读全文方法、实验与附录，未执行实验。
- **通读：** 未完成
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

## 待补

需通读论文图表、正文和附录并逐项校验结果；当前内容为原始摘要及资讯整理，不替代原文。
