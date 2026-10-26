# PhysWord（arXiv:2610.26854）

> 来源归档（paper）

- **标题：** Physical Words: A Vocabulary for Structuring Latent Reasoning in Robot Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.26854>
- **PDF：** <https://arxiv.org/pdf/2610.26854>
- **作者：** Jiaming Liu, Xinghao Zhu, Yiming Zhang, Shanghang Zhang, Yinxi Wang, Yichu Yang, Zhuoyang Liu, Dantong Niu, Soha Pouya, Juekun Li, Yan Chang
- **arXiv 评论栏：** （无）
- **外链：** <https://nvidia-isaac.github.io/video_to_data/physicalword/>
- **入库日期：** 2026-10-26
- **一句话说明：** 物理词汇：用于组织机器人操作潜在推理的词汇体系。

## 开源状态

- arXiv 摘要页/评论给出项目页或外链（https://nvidia-isaac.github.io/video_to_data/physicalword/）；未发现代码仓库链接（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
2. **周报所述问题：** 语言推理能够描述机器人任务的语义目标，但难以表达细粒度物理变化；潜在状态推理能够预测未来，却缺少可复用的语义结构。
3. **周报所述创新点：** 提出 PhysWord，以原子动作词汇作为语义锚点，将动作词的隐藏状态与未来观测提取的潜在表示对齐。模型自回归预测未来的 Physical Words，并通过视觉或触觉条件表达不同操作阶段的状态变化。在 UniVTAC 和 LIBERO-Long 上分别达到 90.3% 和 99.6% 成功率，并在真实双臂夹爪与灵巧手任务中验证。
4. **arXiv 摘要（英文节选）：** Multimodal reasoning provides robotic foundation models with predictive context needed for reliable action generation. However, language-based reasoning captures task semantics but struggles to express fine-grained physical dynamics, while latent reasoning models future state evolution but offers limited structure for organizing these dynamics into reusable reasoning primitives. To bridge this gap, we introduce PhysWord, a reasoning-before-acting foundation model built on Physical Words, which use atomic action verbs as shared semantic anchors for continuous representations of physical dynamics.

**对 wiki 的映射**

- [paper-physical-words-latent-reasoning-manipulation](../../wiki/entities/paper-physical-words-latent-reasoning-manipulation.md)
