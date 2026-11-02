# ProLoop（arXiv:2610.39319）

> 来源归档（paper）

- **标题：** ProLoop: Progress-Aware Closed-Loop Framework via Structured State Relations for Long-Horizon Robotic Task Planning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.39319>
- **PDF：** <https://arxiv.org/pdf/2610.39319>
- **作者：** Zou, Runxin, Liang, Qingmiao, Zhang, Jiatao, Lv, Xinyan, Qie, Liangyu, Bu, Meiyi, Song, Wei, Zhu, Shiqiang
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** ProLoop：基于结构化状态关系的进度感知长程机器人闭环规划。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 长程操作通常分解为多个子目标，但 LLM 规划器对任务进度的判断主要依赖历史文本，容易误判某个子任务是否真正完成。
3. **周报所述创新点：** 提出 State-Relation Monitor，根据目标状态和实际执行状态之间的结构化关系，判断任务处于执行中、偏离目标还是已经完成。在 AgentBoard ALFWorld 和真实机器人实验中，整体成功率提升 **12.94%**，环境交互次数减少 **32.3%**。
4. **arXiv 摘要（英文节选）：** Long-horizon robotic tasks are commonly decomposed into semantic subgoals to simplify planning. However, subgoal progress in LLM-based planning is typically inferred implicitly from interaction histories. Inaccurate estimates can misguide execution and allow local errors to propagate.

**对 wiki 的映射**

- [paper-proloop-progress-aware-closed-loop](../../wiki/entities/paper-proloop-progress-aware-closed-loop.md)
