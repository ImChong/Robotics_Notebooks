# ManuFacet-1K（Hugging Face）

> 来源归档（dataset）

- **标题：** ManuFacet-1K
- **类型：** dataset / contact-rich / force-synchronized / precision-assembly
- **Hugging Face：** <https://huggingface.co/datasets/Pinelab/ManuFacet-1K>
- **组织：** Pinelab（NTU PINE Lab）
- **论文：** <https://arxiv.org/abs/2609.01596>
- **项目页：** <https://pine-lab-ntu.github.io/facet-0/>
- **入库日期：** 2026-09-03
- **一句话说明：** 约 1000 小时力同步精密装配语料（示范 + 闭环 rollout）；公开卡目前暴露 `Facet0-1/` 发布。

## 访问要点（截至 2026-09-03）

| 项 | 内容 |
|----|------|
| 发布 | `Facet0-1/` **READY** |
| 布局 | 每发布含 `data/` `meta/` `videos/` |
| 体积 | 卡片约 **25.2 GB** / **7,866** rows |
| 论文口径 | ~1000 h；CPU 37.3% / Disk 23.4% / RAM 21.9% / GPU 17.3%；三本体（UR7e / xArm / Franka） |
| 帧状态 | 13 维：末端位姿 6 + 夹爪 1 + 腕部 wrench 6；训练时间线 15 Hz，力环 200 Hz |
| 相标签 | approach / align / insert / press / seat / fasten / retreat |

论文称另含数据卡与策展代码；**训练代码仓库仍为占位**，见 [`pine-lab-ntu-facet`](../repos/pine-lab-ntu-facet.md)。

## 关联资料

- 论文：[`sources/papers/facet_0_arxiv_2609_01596.md`](../papers/facet_0_arxiv_2609_01596.md)
- 项目页：[`sources/sites/facet-0.md`](../sites/facet-0.md)
- 代码占位：[`sources/repos/pine-lab-ntu-facet.md`](../repos/pine-lab-ntu-facet.md)
- Wiki：[`wiki/entities/paper-facet-0.md`](../../wiki/entities/paper-facet-0.md)
