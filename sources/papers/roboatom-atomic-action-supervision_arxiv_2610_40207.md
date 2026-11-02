# RoboAtom（arXiv:2610.40207）

> 来源归档（paper）

- **标题：** RoboAtom: Large-scale Atomic-Action Supervision for Robot Learning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.40207>
- **PDF：** <https://arxiv.org/pdf/2610.40207>
- **作者：** Wei, Xueyuan, Wang, Xuanxu, Li, Xudong, He, Yuping, Gan, Wanshui, Peng, Jiaqi, Guo, Wenqi, Zhou, Jianjun et al.
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** RoboAtom：面向机器人学习的大规模原子动作监督。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 大规模机器人数据通常只包含整段任务指令，缺少细粒度动作语义和任务阶段标注，不利于策略学习动作组合与长程执行。
3. **周报所述创新点：** 构建 RoboAtom-Data 和 RoboAtom-Bench，使用原子动作表示组织机器人操作数据。结合人工标注、自动标注和同任务扩展机制生成细粒度监督，并研究这些标注如何用于机器人领域预训练、监督微调和策略执行。
4. **arXiv 摘要（英文节选）：** Fine-grained action annotations can support skill reuse and compositional generalization in robot policies. However, robot datasets typically provide task-level instructions without subtask annotations. Although some include subtask labels, inconsistent formats and conventions hinder cross-source joint training.

**对 wiki 的映射**

- [paper-roboatom-atomic-action-supervision](../../wiki/entities/paper-roboatom-atomic-action-supervision.md)
