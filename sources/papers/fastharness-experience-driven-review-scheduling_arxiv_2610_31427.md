# FastHarness（arXiv:2610.31427）

> 来源归档（paper）

- **标题：** FastHarness: Experience-Driven Review Scheduling for Vision-Language-Action Policies
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.31427>
- **PDF：** <https://arxiv.org/pdf/2610.31427>
- **作者：** Tao Lin, Yuxin Du, Haoxiao Wang, Zile Zhong, Bo Zhao
- **arXiv 评论栏：** Github:this https URL
- **代码：** <https://github.com/FastHarness/fast-harness>；[仓库归档](../repos/fastharness.md)
- **项目页：** <https://fastharness.github.io/>；[站点归档](../sites/fastharness.md)
- **入库日期：** 2026-10-26
- **一句话说明：** FastHarness：面向视觉语言动作策略的经验驱动复查调度。

## 开源状态

- 2026-10-27 核查：官方 GitHub 仓库公开代码，MIT 许可证；RoboDojo 与策略服务器需自行提供。项目页本次无法访问，数据开放情况未核实。

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
2. **周报所述问题：** 长程机器人执行过程中，需要适时检查任务状态。频繁检查会增加计算开销，检查过少又可能导致错误积累。
3. **周报所述创新点：** FastHarness 将历史执行经验引入复查调度，研究如何根据经验决定何时重新评估 VLA 的执行状态。
4. **arXiv 摘要（英文节选）：** Agentic robot systems combine language-model reasoning with robot policies to monitor execution and recover from failures. However, reviewing every action chunk incurs substantial overhead, even in familiar situations. We introduce FastHarness, an experience-driven execution harness that adaptively schedules GPT-6 Astra supervision of a {\pi}0.5 policy.

**对 wiki 的映射**

- [paper-fastharness-experience-driven-review-scheduling](../../wiki/entities/paper-fastharness-experience-driven-review-scheduling.md)
