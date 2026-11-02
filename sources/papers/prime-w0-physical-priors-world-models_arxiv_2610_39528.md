# PRIME W0（arXiv:2610.39528）

> 来源归档（paper）

- **标题：** PRIME W0: Physical Priors from World Models for Embodied Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.39528>
- **PDF：** <https://arxiv.org/pdf/2610.39528>
- **作者：** Peng, Huabin, Jin, Fagang, Wei, Yancong, Ma, Tengfei, Liu, Chuhang, Yu, Chao, Chen, Liqing
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** PRIME W0：利用世界模型物理先验进行具身操作。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** VLA 通常只使用当前观测生成动作，缺少长期历史信息和未来环境变化的有效条件化。
3. **周报所述创新点：** 提出分层条件 Flow Matching 框架，结合 VLM 逐层 KV 注入、自适应 MoE 动作专家和异步世界模型子目标图像预测。同时引入历史视觉特征桥接和协方差结构噪声，使 Flow Matching 的动作生成过程能够利用机器人运动维度之间的耦合关系。
4. **arXiv 摘要（英文节选）：** Vision-Language-Action (VLA) models excel at instruction following, yet most remain feed-forward: they condition actions only on the current observation and discard both long-horizon temporal context and the hierarchical structure of VLM representations. We present PRIME W0, a hierarchical conditional flow-matching framework that closes this gap through three synergistic designs: (1)per-layer KV injection from a pretrained VLM into an Adaptive MoE action expert; (2)dynamics-aware conditioning via asynchronous world-model subgoal-image prediction and a DINOv2 historical-frame semantic bridge; and (3)Action Dynamic Manifold (ADM) flow matching with covariance-structured noise that respects robot kinematic coupling. PRIME W0 is trained in four stages---VLM pre-training, DAT/NTP visuomotor pre-training, DiT flow-matching decoder assembly with WM injection, and end-to-end SFT---and achieves state-of-the-art results on LIBERO and real-robot benchmarks, with systematic ablations validating each architectural component.

**对 wiki 的映射**

- [paper-prime-w0-physical-priors-world-models](../../wiki/entities/paper-prime-w0-physical-priors-world-models.md)
