# PEAR（Pixel-Talk 官方仓库）

> 来源归档

- **标题：** PEAR — Pixel-aligned Expressive humAn mesh Recovery
- **类型：** repo
- **组织：** Pixel-Talk / IDEA
- **代码：** <https://github.com/Pixel-Talk/PEAR>
- **论文：** <https://arxiv.org/abs/2601.22693>
- **项目页：** <https://wujh2001.github.io/PEAR/>
- **入库日期：** 2026-07-11
- **一句话说明：** SIGGRAPH 2026 官方实现：单 ViT-B 单图回归 EHM-s（SMPL-X + scaled FLAME），>100 FPS 推理；训练分两阶段（参数监督 + GUAVA 像素精炼）；数据集待发布。
- **沉淀到 wiki：** [PEAR](../../wiki/entities/paper-pear-pixel-aligned-expressive-hmr.md)

---

## 仓库要点（README ingest 快照）

| 项 | 说明 |
|----|------|
| 会议 | SIGGRAPH 2026 |
| 输入 | 单张 RGB，256×256，**无需**脸/手/人体裁剪 |
| 输出 | SMPL-X 身体/手 + FLAME 头（EHM-s）+ 相机 |
| 骨干 | ViT-B/16 统一 encoder–decoder |
| 速度 | 公开页报告 **>100 FPS**（4090/L40S 级） |
| 训练 | Stage1 参数+关键点；Stage2 GUAVA 光度损失 |
| 数据 | 模块化伪标签（ProHMR/HAMER/TEASER/DWPose）；**Dataset coming soon** |
| Stars | 社区关注度较高（2026-07 约 270+） |

## 对 wiki 的映射

- 主实体页：**`wiki/entities/paper-pear-pixel-aligned-expressive-hmr.md`**
- 论文摘录：**`sources/papers/pear_arxiv_2601_22693.md`**
- 项目页：**`sources/sites/pear-wujh2001-github-io.md`**
