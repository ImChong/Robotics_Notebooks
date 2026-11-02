# HiFOR（arXiv:2610.40373）

> 来源归档（paper）

- **标题：** HiFOR: Hierarchical Foresight with Action-Guided World Grounding for Vision-Language-Action Model
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.40373>
- **PDF：** <https://arxiv.org/pdf/2610.40373>
- **作者：** Ding, Rongxing, Qu, Hongyu, Wu, Kehong, Wang, Xudong, Zhang, Fan, Xiang, Xinguang, Shu, Xiangbo
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** HiFOR：面向视觉语言动作模型的分层未来预测与动作引导世界信息关联。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** VLA 引入未来状态预测后，短期动作变化与长期任务目标通常分别建模。预测得到的大量未来信息也不一定与当前动作有关。
3. **周报所述创新点：** 提出分层未来预测框架，同时预测短期状态、语义里程碑和最终目标。通过结构化潜在世界表示组织不同时间尺度的信息，再由动作生成过程动态检索相关未来特征，减少无关预测信息对动作决策的干扰。
4. **arXiv 摘要（英文节选）：** Vision-Language-Action (VLA) models have demonstrated promising capabilities in robotic manipulation by integrating visual perception, language understanding, and action generation. Recent advances have further incorporated future prediction to enhance the understanding of environmental dynamics and task progression. However, existing methods often model short-term dynamics and long-term task objectives separately, while directly incorporating predicted futures into action policies without sufficiently considering their relevance to current action requirements.

**对 wiki 的映射**

- [paper-hifor-hierarchical-foresight-vla](../../wiki/entities/paper-hifor-hierarchical-foresight-vla.md)
