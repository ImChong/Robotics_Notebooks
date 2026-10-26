# Recon-Augmented Bimanual Pretrain（arXiv:2610.29716）

> 来源归档（paper）

- **标题：** Reconstruction-Augmented Visuomotor Pre-Training of 3D-Grounded Bimanual Dexterity from RGB-Only Human Videos
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.29716>
- **PDF：** <https://arxiv.org/pdf/2610.29716>
- **作者：** Xingyi He, Zhiwen Qiu, Adhitya Polavaram, Yunhao Cao, Cory Fan, Julius Samwer, Saleh Kalantari, Kuan Fang
- **arXiv 评论栏：** Project page: this https URL
- **外链：** <https://spatial-dex.github.io/>
- **入库日期：** 2026-10-26
- **一句话说明：** 利用三维重建增强的视觉运动预训练，从纯 RGB 人类视频学习双臂灵巧操作。

## 开源状态

- arXiv 摘要页/评论给出项目页或外链（https://spatial-dex.github.io/）；未发现代码仓库链接（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
2. **周报所述问题：** 人类 RGB 视频缺少机器人动作标签和直接可用的三维操作状态，限制了双臂策略从大规模人类视频中学习。
3. **周报所述创新点：** 将三维重建引入视觉运动预训练，使模型能够利用纯 RGB 视频中的三维操作信息学习双臂灵巧能力。
4. **arXiv 摘要（英文节选）：** We present SpatialDex, a framework that learns 3D-grounded bimanual dexterity from RGB-only human videos through reconstruction-augmented visuomotor pre-training. The central idea is to learn human behavior together with its spatial context. From each video, we reconstruct the scene geometry and estimate the hand motion, align them at a shared video-level scale using the hand-size prior of 3D hand estimates.

**对 wiki 的映射**

- [paper-reconstruction-augmented-bimanual-pretraining-rgb-human-video](../../wiki/entities/paper-reconstruction-augmented-bimanual-pretraining-rgb-human-video.md)
