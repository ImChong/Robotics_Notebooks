# PixVid 官方项目页

- **标题：** Back to Pixels: Lifting Latent Video Diffusion into Pixel Space
- **类型：** site
- **项目页：** https://chenshuo20.github.io/PixVid/
- **论文：** https://arxiv.org/abs/2611.17336
- **HTML：** https://arxiv.org/html/2611.17336v1
- **代码：** 截至 2026-11-17 未找到可访问的官方 GitHub 仓库地址；官网页首虽有 GitHub / Hugging Face 字样，但未列出可点击发布链接
- **核对日期：** 2026-11-17
- **开源状态：** 代码和权重的公开发布待确认；不要将页首文字标签当成已开源证明
- **核对范围：** Overview、Method（像素迁移 / 高分辨率 / 因果 detailer）、Low-latency autoregressive generation、Results、BibTeX 与资源链接区
- **通读：** 已完成
- **沉淀到 wiki：** [PixVid 论文实体](../../wiki/entities/paper-pixvid-back-to-pixels.md)

## 官方展示要点

- 像素 patch token 与因果卷积 detailer 替代 VAE；采用逐帧条件生成。
- 在 960p 使用与 480p 相同的 30×52 每帧 token 网格，提升分辨率主要通过增大 patch 和增加解码上采样层。
- 每帧控制输入可在下一帧生效；对比展示的传统 temporally compressed latent 模型按帧块响应。
- 项目视频、论文评测数字属于作者展示，尚无本次运行验证。
