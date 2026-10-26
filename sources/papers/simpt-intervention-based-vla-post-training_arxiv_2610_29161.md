# SimPT（arXiv:2610.29161）

> 来源归档（paper）

- **标题：** SimPT: An Automated Simulation Framework for Intervention-Based VLA Post-Training
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.29161>
- **PDF：** <https://arxiv.org/pdf/2610.29161>
- **作者：** Caicheng Wang, Zipeng Ye, Zhexuan Zhou, Zilong Huang, Xi Zhang, Yuchen Xie
- **arXiv 评论栏：** 11 pages, 6 figures

- **入库日期：** 2026-10-26
- **一句话说明：** SimPT：面向干预式 VLA 后训练的自动化仿真框架。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
2. **周报所述问题：** VLA 在部署过程中需要通过失败纠正提高性能，但真实机器人上的人工干预数据采集成本较高。
3. **周报所述创新点：** SimPT 将干预式后训练流程引入自动化仿真环境，研究通过仿真中的执行、干预与策略更新降低后训练的数据采集成本。
4. **arXiv 摘要（英文节选）：** Human-in-the-loop (HIL) post-training improves vision-language-action (VLA) policies through expert corrections, but costly human supervision and repeated real-robot evaluation limit algorithm exploration and development. We introduce SimPT (Simulation-based Post-Training), a simulation framework that automates intervention-based VLA post-training by integrating policy rollouts, corrective data collection, policy training, and standalone evaluation. To automatically determine intervention timing and recovery strategies, we develop an Agent-Compiled Progress Graph that tracks task progress using privileged simulator state and retains stable snapshots during policy execution.

**对 wiki 的映射**

- [paper-simpt-intervention-based-vla-post-training](../../wiki/entities/paper-simpt-intervention-based-vla-post-training.md)
