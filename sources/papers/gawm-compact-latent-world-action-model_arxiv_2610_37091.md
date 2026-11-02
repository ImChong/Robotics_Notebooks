# GAWM（arXiv:2610.37091）

> 来源归档（paper）

- **标题：** GAWM: A Compact Latent World-Action Model for Multi-Task Robot Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.37091>
- **PDF：** <https://arxiv.org/pdf/2610.37091>
- **作者：** Liu, Yang, Liu, Xiaoli, Li, Liuzhao, Song, Yuduo, Liu, Muyuan, Gao, Xiang
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** GAWM：面向多任务机器人操作的紧凑潜在世界动作模型。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 大型 WAM 通常依赖数十亿参数的视觉语言或视频生成模型，训练成本和推理延迟较高。
3. **周报所述创新点：** 使用冻结的 DINOv3 Encoder，将每个相机视角压缩成少量 Token。轻量预测器根据任务条件预测两个未来时间尺度的视觉 Latent，再由动作 Decoder 生成动作块。模型参数量低于 100M，无需机器人或视频预训练，在 RoboTwin 2.0 上超过论文比较的部分十亿参数 WAM。
4. **arXiv 摘要（英文节选）：** World-action models (WAMs) incorporate the prediction of future observations into robot policy learning. However, most WAMs build on billion-parameter vision-language or video-generation backbones, which incur high training cost and inference latency. In this letter, we present GAWM (Geometry-Aware World Model), a compact multi-task WAM that predicts future observations in the latent space of a frozen DINOv3 encoder.

**对 wiki 的映射**

- [paper-gawm-compact-latent-world-action-model](../../wiki/entities/paper-gawm-compact-latent-world-action-model.md)
