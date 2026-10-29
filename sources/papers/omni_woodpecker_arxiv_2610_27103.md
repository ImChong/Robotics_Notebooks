# Omni-Woodpecker: Hallucination Correction via Evidence-Prior Rebalancing for Omni LLMs

> 来源归档（paper）

- **论文：** <https://arxiv.org/abs/2610.27103>
- **Hugging Face：** <https://huggingface.co/papers/2610.27103>
- **代码：** <https://github.com/ZiiJianLiu/Omni-Woodpecker>（[仓库归档](../repos/omni-woodpecker.md)）
- **作者：** Zijian Liu、Chaoyou Fu、Yan Yang、Caifeng Shan、Ran He
- **机构：** 南京大学；中国科学院大学
- **发表日期：** 2026-10-19（arXiv v1）；**入库日期：** 2026-10-28

## 摘录 1：问题与方法

音视频融合表征中，接地证据与模型先验可能相互竞争。OWP 使用冻结模型的互补模态视图识别冲突，以正交分解区分证据和先验，并在解码前自适应调整表征；该方法无需训练。

## 摘录 2：幻觉基准

官方结果显示，Qwen2.5-Omni-7B 在 CMM 和 AVHBench 的准确率分别由 73.2、75.1 提升至 86.1、83.7；VideoLLaMA2-AV 分别由 76.2、70.2 提升至 80.8、74.1。CMM 感知准确率则分别从 89.5、87.2 降至 86.7、83.8。

## 摘录 3：通用任务与适用边界

Qwen 在 OmniBench、WorldSense、MUSIC-AVQA、AVQA 的提升分别为 1.93、2.25、1.00、0.67 个百分点。论文实验针对音视频问答和两种骨干，不能直接推断机器人闭环控制收益。

**对 wiki 的映射：** [Omni-Woodpecker 实体页](../../wiki/entities/paper-omni-woodpecker.md)。
