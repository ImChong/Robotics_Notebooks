# 可训练 CPG（arXiv:2610.36537）

> 来源归档（paper）

- **标题：** A Trainable CPG Framework for Quadruped Locomotion
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.36537>
- **PDF：** <https://arxiv.org/pdf/2610.36537>
- **作者：** Liu, Xiyan, Liu, Yide, Mao, Guoyong, Qu, Shaoxing
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 面向四足运动的可训练中央模式发生器框架。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】人形／四足33篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
2. **周报所述问题：** 多步态控制通常为每种步态和转换分别训练策略，难以获得可解释且连续的统一控制。
3. **周报所述创新点：** 八神经元对称 CPG 生成节律，共享腿部执行网络将节律与轨迹参数转换为关节参考；一个反馈策略只调节六个物理可解释变量，即可实现五种步态及全部方向转换。
4. **arXiv 摘要（英文节选）：** We present CPG-ENN, a trainable quadruped locomotion framework that combines explicit neural coordination with learned motion realization. A symmetry-based eight-neuron central pattern generator (CPG) supplies gait rhythms, while a pretrained, leg-shared execution neural network (ENN) maps rhythmic signals and trajectory parameters to joint-angle references without online inverse kinematics. A single feedback policy modulates six physically interpretable trajectory variables.

**对 wiki 的映射**

- [paper-trainable-cpg-quadruped-locomotion](../../wiki/entities/paper-trainable-cpg-quadruped-locomotion.md)
