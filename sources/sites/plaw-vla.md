# PLaW-VLA 项目页

- **类型**：项目静态站点
- **收录日期**：2026-10-12
- **站点**：<https://rainyrobo.github.io/PLaW-VLA/>
- **论文**：<https://arxiv.org/abs/2610.12285>
- **代码：** <https://github.com/RainyRobo/PLaW-VLA>
- **视频：** <https://www.youtube.com/watch?v=k0GPcDAwq5g>
- **沉淀到 wiki：** [paper-plaw-vla-predictive-latent-world-model-vla.md](../../wiki/entities/paper-plaw-vla-predictive-latent-world-model-vla.md)

## 开源核查（2026-10-12）

| 项 | 结论 |
|----|------|
| 代码 | **已开源**（Apache-2.0）：训练 Stage I–III、LIBERO 评测、策略服务；详见 [sources/repos/plaw_vla.md](../repos/plaw_vla.md) |
| 权重 | 页面有 Hugging Face 图标但未列 URL；仓库 README 写明官方 PLaW-VLA 预训练检查点「计划发布 / Pending」 |
| 数据 | LIBERO EEF 数据集 `RainyBot/libero_v3_eef` 已在 HF；论文预训练混合依赖作者核验清单 |

## 站点摘录要点

- 作者单位：吉林大学、Astribot、哈尔滨工业大学（深圳）、香港理工大学、东京大学。
- 真机成功率表（π0 / π0.5 / PLaW-VLA）：Fold Towel 50/60/80，Play Basketball 47.5/42.5/50，Pack Toy 46.7/50/66.7，Organize Stationery 43.3/41.7/55，另有两个零样本变体行。
- BibTeX 标注 CoRL 2026。
