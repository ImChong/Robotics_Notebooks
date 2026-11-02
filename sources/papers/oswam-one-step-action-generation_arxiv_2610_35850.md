# OSWAM（arXiv:2610.35850）

> 来源归档（paper）

- **标题：** OSWAM: An Effective World Action Model with One-Step Action Generation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.35850>
- **PDF：** <https://arxiv.org/pdf/2610.35850>
- **作者：** Zhang, Guowen, Chen, Liyi, Hu, Xu, Wang, Pengfei, Chen, Jingfan, He, Chenhang, Zhang, Lei
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** OSWAM：具有单步动作生成能力的高效世界动作模型。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** WAM 依赖多步去噪生成动作，推理延迟较高。直接减少去噪步数又容易损害动作质量。
3. **周报所述创新点：** 引入 Imagination Tokens，为扩散过程提供结构化动作先验，并通过特权信息蒸馏学习未来条件信息。模型只需一步扩散即可生成动作，相比直接策略式 WAM 约加速 3 倍；在 RTX 5090 上编译推理延迟最低约 40 ms。
4. **arXiv 摘要（英文节选）：** Direct-policy World Action Models (WAM) have demonstrated the benefits of decoupling action generation from explicit future visual prediction for efficient inference in autonomous driving and embodied control. However, they still require tens of denoising steps to achieve competitive action quality, incurring substantial latency and hindering real-time deployment. Simply reducing the number of denoising steps will collapse the nonlinear multi-step denoising trajectory into coarse updates, substantially degrading action quality.

**对 wiki 的映射**

- [paper-oswam-one-step-action-generation](../../wiki/entities/paper-oswam-one-step-action-generation.md)
