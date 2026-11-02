# JEDI（arXiv:2610.40356）

> 来源归档（paper）

- **标题：** JEDI: Joint-Embedding Demonstration Compression for In-Context Imitation Learning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.40356>
- **PDF：** <https://arxiv.org/pdf/2610.40356>
- **作者：** Liang, Jiazhao, He, Zhewen, Yuan, Shuaihang, Huang, Hao, Fang, Yi
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** JEDI：面向上下文模仿学习的联合嵌入示范压缩。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 上下文模仿学习需要将示范轨迹作为条件输入，但完整示范包含大量图像和动作 Token，限制了可同时使用的示范数量。
3. **周报所述创新点：** 将每条示范压缩为少量 Latent Slots，并使用冻结的 V-JEPA 预测查询轨迹下一阶段的关键帧表示。同时引入任务和执行过程的反事实示范，使压缩表示保留抓取位置、操作顺序等信息，而不只是任务语义。还通过预测误差选择更有用的示范，并支持人手视频作为机器人操作条件。
4. **arXiv 摘要（英文节选）：** In-context imitation learning (ICIL) adapts robot policies through test-time demonstrations, but long multimodal trajectories constrain how many examples fit in context. We ask what a compressed demonstration should preserve. We propose JEDI, which compresses demonstrations into latent slots trained to predict the query episode's next-stage keyframe in a frozen video-JEPA space.

**对 wiki 的映射**

- [paper-jedi-demonstration-compression-in-context-il](../../wiki/entities/paper-jedi-demonstration-compression-in-context-il.md)
