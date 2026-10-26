# Critic-Weighted Residual Flow（arXiv:2610.29117）

> 来源归档（paper）

- **标题：** Learning to Absorb Corrections: Critic-Weighted Residual Flow for Vision-Based Insertion
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.29117>
- **PDF：** <https://arxiv.org/pdf/2610.29117>
- **作者：** Yiqing Yuan, Zhi Li, Jinrong Yang, Mingshan Sun, Mei Yu Zhi, Yong Zhao, Liangliang Ren, Hui Cheng
- **arXiv 评论栏：** 9 pages, 7 figures

- **入库日期：** 2026-10-26
- **一句话说明：** 学习吸收纠正：面向视觉插入任务的 Critic 加权残差 Flow。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
2. **周报所述问题：** 视觉插入任务需要较高位置精度。基础策略执行过程中出现的误差，需要通过纠正动作进行补偿，但不同纠正动作的价值并不相同。
3. **周报所述创新点：** 将 Critic 加权与残差 Flow 结合，研究利用纠正反馈优化视觉插入策略，使策略能够学习更有效的误差补偿行为。
4. **arXiv 摘要（英文节选）：** Vision-based insertion requires correcting alignment errors while retaining useful behavior learned from demonstrations. We present ACT--Flow, an online learning framework that couples an adaptive Action Chunking Transformer (ACT) with a conditional residual flow policy. Following anchor-guided warmup, ACT absorbs successful executed actions through behavioral cloning, while residual targets are recomputed relative to the updated prior.

**对 wiki 的映射**

- [paper-critic-weighted-residual-flow-insertion](../../wiki/entities/paper-critic-weighted-residual-flow-insertion.md)
