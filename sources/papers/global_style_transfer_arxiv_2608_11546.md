# Through Van Gogh’s Eyes: Global Style Transfer with Diffusion Model

> 来源归档：论文，arXiv:2608.11546v2（2026-08-13），ECCV 2026。

- **作者：** Jeongha Lee、Yujin Kim、Ghazanfar Ali、Suhyun Kim、Jae-In Hwang。
- **原文：** https://arxiv.org/abs/2608.11546v2
- **可访问的全文：** https://rt.http3.lol/index.php?q=aHR0cHM6Ly9hcnhpdi5vcmcvaHRtbC8yNjA4LjExNTQ2djIjYmliLmJpYjc
- **作者介绍与代码状态：** https://ghazanfarali.com/research/global-style-transfer/ （截至 2026-11-05 未列公开代码或权重，原实现由机构持有）
- **核对范围：** v2 正文方法、算法、实验与可访问附录；实验解释以论文表格为准。
- **通读：** 已完成

## 主要摘录

1. **问题：** 单张参考图的迁移不等于艺术家整体风格；艺术家名字提示可能偏向少数代表作。GST 从同一艺术家的多幅画作学习多对一的风格表示。
2. **GSG：** 固定提示词“A painting”，用噪声重建训练轻量的 Style Extraction Function，学习 U-Net 瓶颈 h-space 残差。对 wiki 的映射：[项目实体](../../wiki/entities/paper-global-style-transfer.md)。
3. **CAG：** 内容图经 DDIM inversion，反向过程用 CLIP 深层感知损失梯度约束内容，同时允许风格化变形。对 wiki 的映射：[项目实体](../../wiki/entities/paper-global-style-transfer.md)。
4. **实验：** WikiArt 风格参考、VanGogh2Photo 内容；Table 2 中梵高实验 GST 的 CLIP-Div 为 0.297、1-Prec 为 0.988，但 FID 9.46 和 ArtFID 19.25 并非该表最优。对 wiki 的映射：[项目实体](../../wiki/entities/paper-global-style-transfer.md)。
5. §2.1–2.2、§3.1、§3.3、表 1、§4.1、表 2、图 5–7、附录（实现细节、图 13）：用于实体页「与其他工作对比」（StyleInjection 等单图迁移、Textual Inversion、DreamBooth+LoRA、Custom Diffusion、StyleAligned、原始 Stable Diffusion、Asyrp）（2026-11-05 补）。
