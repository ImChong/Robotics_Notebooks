# LAC-WM 项目页（lacwm.github.io）

- **类型**：项目静态站点
- **收录日期**：2026-10-12
- **站点**：<https://lacwm.github.io/>
- **论文**：<https://arxiv.org/abs/2610.10846>（站点另链 OpenReview PDF：<https://openreview.net/pdf?id=S30Y1sBz34>）
- **沉淀到 wiki：** [paper-lac-wm-cross-embodiment-latent-action-world-model.md](../../wiki/entities/paper-lac-wm-cross-embodiment-latent-action-world-model.md)

## 开源核查（2026-10-12）

| 项 | 结论 |
|----|------|
| 代码 | **未发布 / 无链接**：抓取页面源码，站内链接仅有锚点、视频、样式表与 OpenReview PDF，没有 GitHub / Hugging Face URL；arXiv 页亦无官方仓库 |
| 权重 / 数据 | 未列出；预训练使用公开数据集 EgoDex、AgiBot、DROID，微调数据为 RoboCasa+MimicGen 自生成 |
| 许可 | 未说明 |

## 站点摘录要点

- 标注 ICML 2026；作者 Stanford，部分工作在 Meta FAIR Robotics 完成。
- 展示 LIBERO（未见物体）：LAC-WM 92.0 / EAC-WM 82.4 / VLA-mean 81.6 / VLA-random 74.4；灵巧操作（双臂 Franka + Allegro 手）：22 / 15 / 17 / 15；潜在动作 UMAP 跨数据集重叠、显式编码器按数据集分离；规划与跨本体预测的演示视频（assets/videos/）。
