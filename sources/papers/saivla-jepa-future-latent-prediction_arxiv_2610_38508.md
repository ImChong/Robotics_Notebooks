# SaiVLA-JEPA（arXiv:2610.38508）

> 来源归档（paper）

- **标题：** SaiVLA-JEPA: Evaluating Future Latent Prediction in JEPA-Conditioned VLA Control
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.38508>
- **PDF：** <https://arxiv.org/pdf/2610.38508>
- **作者：** Xiao, Rufeng, Xiao, Ruihao, Chen, Wei, Sun, Xinhai
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** SaiVLA-JEPA：评估 JEPA 条件化 VLA 控制中的未来潜在表示预测。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** JEPA 可以学习未来视觉表示，但增加未来预测监督是否一定能改善 VLA 的闭环操作性能，仍需要直接验证。
3. **周报所述创新点：** 将冻结的 V-JEPA 特征引入 Flow-Based VLA，并增加两个训练期未来特征预测 Head。推理时移除辅助 Head，保持原有动作采样流程。实验发现，当前 JEPA 特征条件化使成功率从 **71.6% 提升至约 82.5%**，但额外的未来预测监督在该训练预算下没有带来可检测的总体收益。
4. **arXiv 摘要（英文节选）：** Joint-embedding predictive architectures (JEPAs) learn to predict representations rather than pixels, offering a compact way to model how visual scenes evolve. We ask whether this predictive objective improves closed-loop control when a vision-language-action (VLA) policy already uses V-JEPA features. We call our JEPA-conditioned flow policy family SaiVLA-JEPA.

**对 wiki 的映射**

- [paper-saivla-jepa-future-latent-prediction](../../wiki/entities/paper-saivla-jepa-future-latent-prediction.md)
