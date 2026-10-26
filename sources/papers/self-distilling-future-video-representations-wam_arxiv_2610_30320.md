# Self-Distilling WAM（arXiv:2610.30320）

> 来源归档（paper）

- **标题：** Self-Distilling Future Video Representations for World Action Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.30320>
- **PDF：** <https://arxiv.org/pdf/2610.30320>
- **作者：** Huiwon Jang, Suhyeok Jang, Hyeonmo Kang, Seungku Kim, Jinwoo Shin, Sihyun Yu
- **arXiv 评论栏：** 27 pages; Project page: this https URL
- **外链：** <https://huiwon-jang.github.io/FRED/>
- **入库日期：** 2026-10-26
- **一句话说明：** 面向世界动作模型的未来视频表示自蒸馏。

## 开源状态

- arXiv 摘要页/评论给出项目页或外链（https://huiwon-jang.github.io/FRED/）；未发现代码仓库链接（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
2. **周报所述问题：** WAM 需要利用未来视觉变化学习动作，但完整视频预测会带来较高计算成本。如何保留未来信息中的控制相关特征，是提高模型效率的重要问题。
3. **周报所述创新点：** 研究未来视频表示的自蒸馏，使模型通过蒸馏学习未来视觉特征，并将这些表示用于世界动作建模。
4. **arXiv 摘要（英文节选）：** World Action Models (WAMs) have emerged as a promising approach to learning generalist robot policies using rich spatiotemporal priors from pre-trained video diffusion models. A recent line of WAMs improves inference efficiency by conditioning a separate action decoder on intermediate future video representations extracted from a single denoising step, avoiding costly iterative denoising required for video generation. This design makes action prediction depend directly on the quality of these future representations.

**对 wiki 的映射**

- [paper-self-distilling-future-video-representations-wam](../../wiki/entities/paper-self-distilling-future-video-representations-wam.md)
