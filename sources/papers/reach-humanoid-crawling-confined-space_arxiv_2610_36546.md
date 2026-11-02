# REACH（arXiv:2610.36546）

> 来源归档（paper）

- **标题：** REACH: An RL-based Hierarchical Humanoid Crawling Framework for Adaptable Confined Space Traversal
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.36546>
- **PDF：** <https://arxiv.org/pdf/2610.36546>
- **作者：** Xu, Shengzhang, Guo, Yuning, Hu, Weiyi, Xu, Yifeng, Li, Zhongyu, Kong, He
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** REACH：面向可适应狭窄空间穿越的分层强化学习人形爬行框架。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】人形／四足33篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
2. **周报所述问题：** 仅降低双足行走高度仍无法穿越极低空间，而且行走与爬行之间存在较大的姿态和接触模式突变。
3. **周报所述创新点：** 分别训练变高度双足策略和爬行策略，以历史条件 latent 编码接触及地形信息；双向转换策略连接两种模式，激光雷达规划器进一步生成三维路径、速度和主动高度指令。
4. **arXiv 摘要（英文节选）：** Humanoid locomotion in confined spaces requires continuous height adaptation across multiple locomotion modes, yet most existing approaches remain limited to height-adjustable bipedal walking. We present REACH, an RL-based hierarchical framework that enables a Unitree G1 humanoid to traverse height-constrained environments through variable-height bipedal walking and crawling. The learned policies regulate body height over complementary command ranges of 1.27--0.50m in bipedal locomotion and 0.50--0.20m in crawling, extending traversal beyond the clearance limits of crouched walking.

**对 wiki 的映射**

- [paper-reach-humanoid-crawling-confined-space](../../wiki/entities/paper-reach-humanoid-crawling-confined-space.md)
