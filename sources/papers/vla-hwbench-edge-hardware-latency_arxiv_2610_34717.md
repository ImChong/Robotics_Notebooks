# VLA-HWBench（arXiv:2610.34717）

> 来源归档（paper）

- **标题：** VLA-HWBench: Where the Time Goes When VLA Models Run on Edge Hardware
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.34717>
- **PDF：** <https://arxiv.org/pdf/2610.34717>
- **作者：** Xu, Borui, Tan, Hongshi, Jiang, Wenqi, He, Bingsheng, Wu, Ming, Zeng, Jason, Heinrich, Michael
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** VLA-HWBench：视觉语言动作模型在端侧硬件上的推理耗时分析。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** VLA 的推理延迟不仅取决于模型参数量，还受到动作生成方式、算子实现、CPU 预处理和内存带宽影响。
3. **周报所述创新点：** 在 NVIDIA Jetson AGX Thor 和 AMD Ryzen AI Max+ PRO 395 上评测 18 个开源 VLA 模型，将推理流程分为七个阶段。实验发现，动作生成方式比参数量更能解释延迟差异；使用较少 Flow Matching 步数生成较长动作块的模型，更容易满足实时控制要求。
4. **arXiv 摘要（英文节选）：** Vision-language-action (VLA) models increasingly run on edge devices aboard robots, where inference latency constrains closed-loop robot control. However, it remains unclear how to design VLA models, their inference software and edge hardware so that robots can run VLA inference in real time. Most VLA models report only task success, while prior studies cover limited action-generation strategies, divide inference into at most three stages, and lack comparisons of inference optimizations on the same models.

**对 wiki 的映射**

- [paper-vla-hwbench-edge-hardware-latency](../../wiki/entities/paper-vla-hwbench-edge-hardware-latency.md)
