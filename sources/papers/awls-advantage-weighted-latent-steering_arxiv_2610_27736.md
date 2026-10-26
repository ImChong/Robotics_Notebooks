# AWLS（arXiv:2610.27736）

> 来源归档（paper）

- **标题：** AWLS: Advantage-Weighted Latent Steering for One-Step Generative Robotic Manipulation Policies
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.27736>
- **PDF：** <https://arxiv.org/pdf/2610.27736>
- **作者：** Jiahong Tang, Jiaju He, Zhiyuan Yang, Ruyang Han, Haiyue Zhu
- **arXiv 评论栏：** Accepted at CoRL 2026
- **外链：** <https://awls-corl2026.github.io/>
- **入库日期：** 2026-10-26
- **一句话说明：** AWLS：面向单步生成式机器人操作策略的优势加权潜在引导。

## 开源状态

- arXiv 摘要页/评论给出项目页或外链（https://awls-corl2026.github.io/）；未发现代码仓库链接（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
2. **周报所述问题：** 单步生成式策略具有较低推理延迟，但策略优化仍需要区分高质量动作与低质量动作。
3. **周报所述创新点：** AWLS 将优势加权与潜在空间引导结合，通过动作价值信息调整生成式策略的潜在表示，研究在保留单步生成效率的同时改善操作性能。论文已被 CoRL 2026 接收。
4. **arXiv 摘要（英文节选）：** Human-collected real-world robotic manipulation demonstrations often contain mixed-quality trajectories, including hesitant, indirect, or inefficient action segments. While diffusion- and flow-based policies can model such multimodal data, standard behavior cloning (BC) assigns equal likelihood to all demonstrated actions and may therefore reproduce suboptimal modes. Offline RL value estimates offer a natural signal for biasing policies toward higher-value behavioral modes, but directly fine-tuning the generative policy or using critic gradients for policy steering can amplify critic errors and induce prior drift.

**对 wiki 的映射**

- [paper-awls-advantage-weighted-latent-steering](../../wiki/entities/paper-awls-advantage-weighted-latent-steering.md)
