# Adversarial Training for Pixel Diffusion

- **类型：** 论文（arXiv cs.CV）
- **论文：** <https://arxiv.org/abs/2609.38170>
- **HTML 正文：** <https://arxiv.org/html/2609.38170v1>
- **作者：** Xin Lin、Zhifei Zhang、Yuqian Zhou、Haitian Zheng、Zhe Lin、Ming-Hsuan Yang、Truong Nguyen
- **机构：** UC San Diego、Adobe Research、UC Merced；第一作者主页注明该工作完成于其 Adobe Research 实习期间
- **提交日期：** 2026-09-29（v1）
- **入库日期：** 2026-10-11
- **官方项目页 / 代码：** 截至入库日，论文与第一作者主页 publication entry 均未列出本工作的项目页或代码链接。DeCo 与 PixelGen 是文中使用的公开基座，不应误认为本论文的官方实现。
- **wiki：** [像素扩散的对抗训练](../../wiki/entities/paper-adversarial-training-pixel-diffusion.md)

## 摘要与问题

像素扩散模型直接生成 RGB，避免自编码器瓶颈，但仍低估自然图像细粒度统计。论文研究将对抗学习作为已收敛像素扩散模型的后训练校正：保留扩散或 Flow Matching 目标，在非高噪声时间步对预测输出加 GAN loss，不改模型结构和采样流程。

## 方法摘录

- 生成器继续优化原扩散 / Flow Matching loss，并添加 hinge adversarial loss；判别器区分真实干净图与模型预测的干净图。
- 预测干净图在高噪声阶段缺乏可靠全局结构，因此用信号比例阈值门控，只在结构相对稳定、细节仍可修正的时段加入 GAN loss。
- 主要像素骨干是 DeCo 与 PixelGen；latent 对照是 PixArt-α 和 SANA。每个 GAN/no-GAN 对照匹配初始 checkpoint、数据、训练 horizon 和评测。
- DeCo 主配置：512²、约 1.1B 参数，BLIP3o-60k，30k 微调步，DINOv2-text 判别器，GAN 权重 0.1、门控 `t≥0.53`；论文还报告 `t≥0.35` 等不同 operating point。

## 关键结果

- DeCo：DPG Score `81.6→83.3`，FID `33.27→28.59`，CMMD `0.836→0.736`，Recall `0.361→0.406`，TOPIQ `0.711→0.768`。
- PixelGen：DPG Score `78.4→80.8`，FID `33.94→33.20`，Recall `0.319→0.403`，TOPIQ `0.755→0.799`。
- 高频带占比：DeCo `2.1%→3.9%`，PixelGen `3.9%→6.4%`；DeCo 幂律斜率 `2.59→2.24`，接近真实图像约 `2.19`。
- 相同数据与步数的 no-GAN SFT、训练集最近邻和 recall 对照用于排查「只是训练更久」、记忆样本或 mode dropping 等解释。
- 感知损失也可增加高频，但匹配实验中带来色彩/对比度域偏移，并损害分布质量与提示词对齐；普通锐化也未获得 GAN 的完整收益。
- 测试的 PixArt-α / SANA latent 配置没有相近的联合质量提升，解码图像的高频功率变化很小。作者把差异与冻结解码器削弱输出空间可达性联系起来。

## 读法与边界

- 论文证明的是**特定像素骨干与训练配方上**的 post-training 可行性，不是 GAN 对所有生成模型都普遍提升。
- 频谱恢复、分布指标、召回和文本对齐需联合阅读；仅看纹理变锐可能误把伪影当作高频恢复。
- latent 结论限于 PixArt-α 和 SANA 的已测配置，输出空间可达性是受到实验支持的机制解释，不应泛化成所有 latent 模型的定论。
- 预训练 DeCo / PixelGen checkpoint 有公开来源，并不意味着本文训练代码也公开；当前未找到这篇工作的专属官方代码入口。

## 对 wiki 的映射

- [论文实体：方法、实验对照、适用边界](../../wiki/entities/paper-adversarial-training-pixel-diffusion.md)
- [Diffusion Model：增加像素空间后训练实例](../../wiki/concepts/diffusion-model.md)

## 参考链接

- 论文 HTML（作者机构、方法、完整实验与附录）：<https://arxiv.org/html/2609.38170v1>
- 第一作者主页（论文列表与项目资源链接核查）：<https://linxin0.github.io/>
