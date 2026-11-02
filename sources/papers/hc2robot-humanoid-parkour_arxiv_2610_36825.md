# HC2Robot（arXiv:2610.36825）

> 来源归档（paper）

- **标题：** HC2Robot: Hyper-Connection Transformers with Predicted Foresight for Multi-Terrain Humanoid Parkour
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.36825>
- **PDF：** <https://arxiv.org/pdf/2610.36825>
- **作者：** Xu, Wenhao, Mai, Yanheng, Fu, Yifei, Zheng, Nan, Zhou, Yanbo, Huang, Zirui, Ling, Mingsheng, Wang, Xinjue et al.
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** HC2Robot：基于超连接 Transformer 与预测前瞻的人形多地形跑酷。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】人形／四足33篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
2. **周报所述问题：** 非对称 Actor–Critic 中，Critic 能够看到特权接触和地形信息，但部署 Actor 只能使用机载观测，导致复杂地形落脚决策存在信息差。
3. **周报所述创新点：** 前瞻模块从可部署观测预测下一次落脚的时间、位置和支撑质量，将特权监督转化为部署 latent；超连接 Transformer 使用多流路由增强时间推理，在楼梯、沟隙、高台和斜坡上平均成功率达到 97.6%。
4. **arXiv 摘要（英文节选）：** Humanoid parkour requires robots to reason about future footholds and whole-body motions over highly discontinuous terrain. We revisit the information asymmetry in asymmetric actor-critic learning for perceptive humanoid locomotion: the critic exploits privileged simulator signals during training, whereas the deployable actor operates solely from onboard perception. We argue that the key is not exposing privileged information to the actor, but learning a deployable approximation of it to narrow the information gap between training and deployment.

**对 wiki 的映射**

- [paper-hc2robot-humanoid-parkour](../../wiki/entities/paper-hc2robot-humanoid-parkour.md)
