# 连续动作世界模型（arXiv:2610.40401）

> 来源归档（paper）

- **标题：** Continuous-Action World Models: Learning Cross-Embodiment Spatiotemporal Dynamics Priors
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.40401>
- **PDF：** <https://arxiv.org/pdf/2610.40401>
- **作者：** Li, Liangyu, Liu, Qingwen
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 连续动作世界模型：学习跨本体时空动力学先验。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 不同机器人具有不同动作维度和控制频率。直接学习离散动作序列，会使策略依赖特定本体与采样频率。
3. **周报所述创新点：** 提出 CoAWM，将动作轨迹表示为连续时间函数。首先通过动作重建和跨采样频率一致性学习共享潜在函数空间，再将该空间连接到世界模型。模型根据视觉、语言和机器人状态预测潜在动作函数，最后通过任意时间查询生成末端执行器动作，使动作表示与具体控制频率解耦。
4. **arXiv 摘要（英文节选）：** We introduce Continuous-Action World Models (CoAWM), a framework for learning cross-embodiment spatiotemporal dynamics priors that support robot action generation across embodiments and execution frequencies. At its core is a shared latent function space that captures the spatial and temporal structure of robot actions. We learn this space through two stages.

**对 wiki 的映射**

- [paper-continuous-action-world-models-cross-embodiment](../../wiki/entities/paper-continuous-action-world-models-cross-embodiment.md)
