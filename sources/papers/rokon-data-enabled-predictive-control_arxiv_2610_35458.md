# RoKON（arXiv:2610.35458）

> 来源归档（paper）

- **标题：** RoKON: Efficient and Adaptive Data-Enabled Predictive Control for Robotic Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.35458>
- **PDF：** <https://arxiv.org/pdf/2610.35458>
- **作者：** Sharma, Akshun, Wei, Zhongqi, Ou, Buqing, Dümbgen, Frederike
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** RoKON：面向机器人操作的高效自适应数据驱动预测控制。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 机器人操作中的摩擦等动力学参数可能随环境变化。依赖固定动力学模型的预测控制难以及时适应这些变化。
3. **周报所述创新点：** 使用高斯核构建非线性数据预测器，并提出阻尼 Gauss–Newton 优化方法提高实时求解效率。通过混合数据缓存同时保留有用的离线经验和新采集的在线数据，使控制器能够适应变化的接触动力学。
4. **arXiv 摘要（英文节选）：** Data-enabled predictive control has recently emerged as a promising control paradigm, enabling a transparent optimal-control formulation without the need to learn an explicit dynamics model. This paper presents a number of extensions paving the path towards its use in real-time robotics tasks. Following recent work, we utilize a nonlinear predictor based on Gaussian kernels to represent the relationship between input and output data.

**对 wiki 的映射**

- [paper-rokon-data-enabled-predictive-control](../../wiki/entities/paper-rokon-data-enabled-predictive-control.md)
