# PanoLOG / G²PS 项目页（insta360-research-team.github.io/GGPS-Website）

> 来源归档

- **标题：** G²PS: Geometry and Gradient-based Partitioning for Panoramic Outdoor Reconstruction（页面亦称 PanoLOG）
- **类型：** site / project-page
- **URL：** <https://insta360-research-team.github.io/GGPS-Website/>
- **论文：** <https://arxiv.org/abs/2607.08769> — 归档见 [`sources/papers/ggps_panolog_arxiv_2607_08769.md`](../papers/ggps_panolog_arxiv_2607_08769.md)
- **代码：** <https://github.com/Insta360-Research-Team/GGPS> — 归档见 [`sources/repos/ggps.md`](../repos/ggps.md)
- **数据集 / 权重：** <https://huggingface.co/Insta360-Research/GGPS>
- **机构：** 影石研究（Insta360 Research）× 中山大学 × 华南理工大学 × 中国科学院大学 × 哈尔滨工程大学 × 武汉大学
- **入库日期：** 2026-07-26
- **一句话说明：** PanoLOG / G²PS 官方项目页：全景户外 3DGS 重建演示、定量对比表、开源入口（论文 / 代码 / 数据集）与 UE 插件预告。

## 公开信息要点（截至 2026-07-26 核查）

| 项 | 状态 |
|----|------|
| **arXiv / Paper** | 已挂：<https://arxiv.org/abs/2607.08769> |
| **Code** | 已链官方仓 <https://github.com/Insta360-Research-Team/GGPS> |
| **Dataset** | 已链 Hugging Face <https://huggingface.co/Insta360-Research/GGPS> |
| **3DGS Plugin & Unreal Engine** | 标注 **Coming soon**（项目页称 UE 5.8 渲染插件 mid-July 2026 档） |
| **开源结论** | **已开源（训练代码 + 部分数据集）**；预训练 `.ply` 与 UE 插件仍待齐 |

### 页面内容摘要

- **定位：** 用 **G²PS** 解决 ERP 全景「全可见性」下块划分退化；两阶段粗到细 **PanoLOG**；发布 **Pano360**。
- **演示：** GT vs Training 视频、定性对比（远景 / 玻璃立面）、消融可视化。
- **定量表：** A1 无人机子场景 NSC/NSK；X5 手持 BAX/NSN；公开集 Ricoh360 / 360Roam。
- **公告：** 「论文、训练代码与数据集已发布；UE 5.8 3DGS 插件即将上线」。
- **BibTeX：** `@article{panolog2026, ... arXiv:2607.08769}`。

## 为何值得保留

- 步骤 2.5 项目页核查主入口：锁定 **Code / Dataset 已挂、权重与 UE 插件未齐** 的边界。
- 比 PDF 更直观的渲染对比与模型体积数字，便于与 CityGaussian / H3DGS / OmniGS 选型对照。
- 同机构全景线可与 [PanoWorld](../../wiki/entities/paper-panoworld-real-world-panoramic-generation.md)（生成）对照本页（重建）。

## 关联资料

- 论文摘录：[`sources/papers/ggps_panolog_arxiv_2607_08769.md`](../papers/ggps_panolog_arxiv_2607_08769.md)
- 代码仓：[`sources/repos/ggps.md`](../repos/ggps.md)
- Wiki 实体：[`wiki/entities/paper-panolog-ggps.md`](../../wiki/entities/paper-panolog-ggps.md)
