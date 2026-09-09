# i3dGS 项目页（repo-sam.inria.fr/nerphys/i3dgs）

> 来源归档（ingest 配套站点）

- **URL：** <https://repo-sam.inria.fr/nerphys/i3dgs/>
- **标题：** Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency
- **机构：** Inria GraphDeco · Université Côte d'Azur · Université de Rennes · EPFL
- **论文：** <https://arxiv.org/abs/2607.14481> — 归档见 [`sources/papers/i3dgs_arxiv_2607_14481.md`](../papers/i3dgs_arxiv_2607_14481.md)
- **代码：** <https://github.com/graphdeco-inria/i3dgs> — [`sources/repos/i3dgs-graphdeco-inria.md`](../repos/i3dgs-graphdeco-inria.md)
- **PDF：** <https://repo-sam.inria.fr/nerphys/i3dgs/i3dgs.pdf>
- **Slides：** <https://repo-sam.inria.fr/nerphys/i3dgs/siggraph2026-talk/>
- **入库日期：** 2026-09-09
- **一句话说明：** SIGGRAPH 2026 官方落地页：乱序 RGB 即时 3DGS 重建 + 全局一致；VPR/共视性图/聚类回环/渐进层级；链到 GraphDeco 官方实现与 OnTheFly 产品。

## 公开信息要点（截至入库日）

| 项 | 状态 |
|----|------|
| **Paper / Slides / BibTeX** | 已发布；SIGGRAPH Conference Papers 2026 |
| **Demo 视频** | 页内 carousel：playroom / bonsai / room / CityWalk 等即时重建与导航 |
| **方法叙事** | 乱序输入 → VPR 位姿 + 共视性图 → 局部 3DGS + 聚类回环 → 层级高斯 |
| **代码** | **已开源** — GitHub `graphdeco-inria/i3dgs` |
| **商业衍生** | 算法为 [OnTheFly](https://onthefly3d.com) 核心技术的公开表述 |
| **结论** | 项目页 + 代码可用于复现与定性对比；许可为研究/评测向 Immediate3DGS license |

## 页面结构速记

1. **Teaser** — 无序图像序列经 place recognition 与 loop closure 得到即时层级高斯重建。
2. **Abstract** — 与 arXiv 一致；强调首个乱序 + 即时 + 全局一致的 3DGS 方案。
3. **Videos** — 优化过程即时反馈、大场景 CityWalk 层级演示。
4. **Funding** — ERC Advanced Grant NERPHYS（101141721）；Adobe / NVIDIA 捐赠；Grid'5000 / GENCI HPC。

## 关联资料

- 论文摘录：[`sources/papers/i3dgs_arxiv_2607_14481.md`](../papers/i3dgs_arxiv_2607_14481.md)
- 代码仓：[`sources/repos/i3dgs-graphdeco-inria.md`](../repos/i3dgs-graphdeco-inria.md)
- Wiki 实体：[`wiki/entities/paper-i3dgs-immediate-3dgs-unordered.md`](../../wiki/entities/paper-i3dgs-immediate-3dgs-unordered.md)
