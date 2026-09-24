# Beyond Future Prediction: Denoising as Generative Adaptation for Robot Control

> 来源：[具身智能小站 · 13 篇盘点](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)（2026-09-24）

## 元数据

- **arXiv：** [2609.28339](https://arxiv.org/abs/2609.28339)
- **PDF：** https://arxiv.org/pdf/2609.28339
- **代码：** https://github.com/xmz111/NowWAM
- **项目页：** https://xmz111.github.io/NowWAM/
- **开源结论（2026-09-24）：** **已开源**

## 核心摘录

- **一句话：** 生成式视觉先验不必预测未来画面；沿完整去噪轨迹适配当前观测即可稳定控制，并显著降 token 与步时。
- **机制：** 当前观测 latent 作为生成式目标与动作专家共享；训练采样 DiT 去噪轨迹上的表征，推理在 σ=0 干净端点读控制表征，无需额外视觉 rollout。
- **指标：** LIBERO-Plus **87.7%**（FLUX2-Klein）；视觉 token **784→392**；步时 **2.85s→1.63s**；RoboCasa 100-shot **64.9%**。

## 对 wiki 的映射

- 实体页：[NowWAM](../../wiki/entities/paper-nowwam.md)
