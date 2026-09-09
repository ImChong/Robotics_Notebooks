# SOMA 项目页（vcai.mpi-inf.mpg.de/projects/SOMA）

- **标题：** SOMA: From Surface Observations to Muscle Anatomy
- **类型：** site / project-page
- **URL：** <https://vcai.mpi-inf.mpg.de/projects/SOMA/>
- **会议 / 期刊：** European Conference on Computer Vision (ECCV) 2026
- **arXiv：** <https://arxiv.org/abs/2606.09246> — 归档见 [`sources/papers/soma_arxiv_2606_09246.md`](../papers/soma_arxiv_2606_09246.md)
- **代码：** <https://github.com/edualvarado/SOMA> — 归档见 [`sources/repos/soma-surface-muscle.md`](../repos/soma-surface-muscle.md)
- **数据集：** <https://gvv-assets.mpi-inf.mpg.de/soma>
- **视频：** <https://youtu.be/4EWyj6Ew9aw>
- **作者单位：** Max Planck Institute for Informatics（Saarland Informatics Campus）；TU Dortmund University
- **入库日期：** 2026-09-09

## 一句话摘要

MPI-INF VCAI 组官方项目页：从 **多视角 RGB / 体表观测** 反演 **个体化肌肉层时空形变**，配套 **SKIM**（Skin-to-Internal Muscle）五被试多层解剖数据集；级联 U-Net 预测肌肉位移与皮肤残余滑动，以生物力学启发的正则替代传统 FEM 仿真。

## 项目页核查（步骤 2.5 · 2026-09-09）

| 核查项 | 结论 |
|--------|------|
| **Paper** | 绿按钮 → [arXiv:2606.09246](https://arxiv.org/abs/2606.09246) |
| **GitHub Repo** | 绿按钮 → [`edualvarado/SOMA`](https://github.com/edualvarado/SOMA)（MIT） |
| **Dataset** | 绿按钮 → [`gvv-assets.mpi-inf.mpg.de/soma`](https://gvv-assets.mpi-inf.mpg.de/soma) |
| **Video** | YouTube `4EWyj6Ew9aw` |
| **开放程度** | **已开源**：完整管线源码（Suit 处理 → 规范模型 → 注册 → Blender 工具 → 训练/评测）+ SKIM 数据独立下载；大体积 checkpoint / Blender 场景不入 git |

- **代码：** <https://github.com/edualvarado/SOMA>
- **数据集：** <https://gvv-assets.mpi-inf.mpg.de/soma>

> **命名消歧：** 本 SOMA 为 MPI-INF「Surface Observations → Muscle Anatomy」，与 NVIDIA [SOMA-X](../../wiki/entities/soma-x.md)（统一参数化人体拓扑）及 [SOMA Retargeter](../../wiki/entities/soma-retargeter.md)（BVH→人形重定向）无关。

## 公开信息要点（项目页归纳）

### Abstract / 问题

- 参数化人体模型（SMPL 系等）通常只建模 **皮肤外表面**，无法反映驱动运动的 **肌肉与软组织**。
- 传统 **FEM** 软组织仿真准确但不可扩展；现有生物力学工具可算肌力/激活，却常 **不建模外观形变**，难以把激活与可观测解剖关联。
- **逆问题：** 从可见体表（皮肤、姿态）恢复 **肌肉形变**；本文称首次从 **多视角 RGB** 尝试该任务。

### Method

- 在 pose-dependent **corrective blendshape** 思想上扩展到 **体素化多层解剖**（皮肤 / 肌肉 / 骨）。
- 给定骨骼姿态，**两级级联非线性 U-Net**：
  1. 肌肉位移场 \(D_{\mathrm{musc}}\) — 驱动肌肉层鼓胀；
  2. 残余偏移 \(D_{\mathrm{res}}\) — 允许皮肤在组织上 **滑动与压缩**，而非刚性跟随肌肉。
- 监督：SKIM **规范空间 marker 残余**；正则：面积归一 Laplacian 平滑、双调和弯曲、边拉伸与切向滑动约束、棱柱体 **体积保持**（近似软组织不可压）。
- 边界形变经预计算 **重心绑定** 传播到高分辨率 **单块肌肉网格**，由姿态直接得解剖动画。

### SKIM 数据集

- **5 被试**；紧身 suit 嵌入 **ArUco marker**。
- **120 相机** markerless mocap 棚 + **140 相机** 静态扫描得皮肤/肌肉/骨架层与 **单块肌肉 mesh** 模板。
- Marker 展开为规范点云、绑定肌肉、跨帧跟踪 → pose-normalized **残余形变场**。
- 总计约 **45 分钟** 多视角录像 + 骨骼姿态、marker 轨迹、可见性 mask、完整多层解剖 GT。

### BibTeX

```bibtex
@inproceedings{alvaradosoma2026,
  author    = {Alvarado, Eduardo and Kim, Emily and Nolte, Gerrit and Runte, Friedemann and Botsch, Mario and Habermann, Marc and Theobalt, Christian},
  title     = {SOMA: From Surface Observations to Muscle Anatomy},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

## 对 wiki 的映射

- 论文实体：[paper-soma-surface-observations-muscle-anatomy](../../wiki/entities/paper-soma-surface-observations-muscle-anatomy.md)
- 交叉：[UMA](../../wiki/entities/paper-uma.md)、[MAMMA](../../wiki/entities/paper-mamma-markerless-motion-capture.md)、[SMPL-X](../../wiki/concepts/smpl-x.md)、[SOMA-X](../../wiki/entities/soma-x.md)（同名不同工作，对比消歧）
