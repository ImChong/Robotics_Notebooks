# FlowVLA 倾倒（arXiv:2610.29692）

> 来源归档（paper）

- **标题：** FlowVLA: Learning Flow-Aware Physical Completion for Quantitative Robotic Pouring
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.29692>
- **PDF：** <https://arxiv.org/pdf/2610.29692>
- **作者：** Liwen Yang, Yutian Zhang, Jiaqi Zhai, Li Yang, Tutian Tang, Qiwei Shan, Yancheng Bi, Qiankuan Shangguan, Yangkun Zhu, Ce Hao, Dong Wei, Qiaojun Yu, Dibo Hou
- **arXiv 评论栏：** 9 pages, 6 figures

- **入库日期：** 2026-10-26
- **一句话说明：** FlowVLA：面向定量机器人倾倒操作的流动感知物理完成状态学习。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
2. **周报所述问题：** 定量倾倒不仅要求机器人完成倾倒动作，还需要判断流体转移是否达到目标数量。传统操作策略难以仅根据机械臂运动判断任务完成程度。
3. **周报所述创新点：** 将流动状态与物理完成程度引入操作学习，使机器人能够根据物体转移过程调整动作，而不是仅执行预先生成的倾倒轨迹。
4. **arXiv 摘要（英文节选）：** During steady pouring, images and robot pose may remain nearly unchanged while transferred mass accumulates, making the return decision ambiguous for a whole-task vision-language-action (VLA) policy. We present FlowVLA, which separates motor generation from physical-completion estimation. A shared $\pi_{0.5}$ action expert learns approach, pour, and return with boundary-aware supervision.

**对 wiki 的映射**

- [paper-flowvla-quantitative-robotic-pouring](../../wiki/entities/paper-flowvla-quantitative-robotic-pouring.md)
