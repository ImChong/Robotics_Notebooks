# circle-group.github.io/research/HeatKernelTextures（HKTex 项目页）

- **标题：** Heat Kernel Textures: the Geodesic Gaussians That Do Not Splat
- **类型：** site / project-page
- **URL：** <https://circle-group.github.io/research/HeatKernelTextures/>
- **配套论文：** [HKTex（arXiv:2609.07557）](https://arxiv.org/abs/2609.07557) — 归档见 [`sources/papers/hktex_eccv_2026_arxiv_2609_07557.md`](../papers/hktex_eccv_2026_arxiv_2609_07557.md)
- **代码：** <https://github.com/circle-group/hktex> — 归档见 [`sources/repos/hktex.md`](../repos/hktex.md)
- **入库日期：** 2026-09-11

## 一句话摘要

帝国理工学院 Circle Group 的 **ECCV 2026 Best Paper + Long Oral** 项目页：用测地线各向异性热核在 mesh 上做无 UV 内在纹理，黎曼优化 + 曲面 densify/prune，接 Mitsuba 可微 PBR；展示 UV 拟合与多视角逆渲染对比 demo。

## 公开信息要点（截至入库日）

- **机构：** Imperial College London；作者 Simone Foti*、Caner Korkmaz*、Stefanos Zafeiriou、Tolga Birdal。
- **荣誉：** 页首标注 **ECCV 2026: Best Paper Award**。
- **摘要要点：** 消除 UV 展开及浪费、接缝、扭曲、重复顶点与分辨率不均；离散黎曼几何上的各向异性热核 = 测地线高斯；优化与 densification 在物体曲面上进行；可接物理渲染，从现有纹理或多视角图像优化。
- **方法可视化：**
  - *Kernel Modulation* — 源点、扩散角、各向异性、尺度、锐度、RGB；
  - *Optimization & Density Control* — Riemannian GD + 动量（digeo）；流形剪枝与 clone/split densify。
- **结果交互：**
  - *Fitting Existing UV-Textures* — 多物体同步旋转 + 多材质切换；对比 GT UV、低分辨率 UV、MLP、InstantNGP 内在场、VTex、ImageGS、HKTex；
  - *Multi-View Inverse Rendering* — 对比 VTex、MLP、NvDiffRec*、HKTex 的新视角与材质分解。
- **代码入口：** 项目页与 arXiv 摘要均指向 `circle-group.github.io/research/HeatKernelTextures` 与 GitHub 仓（非「即将开源」占位）。

## 为何值得保留

- **步骤 2.5 证据：** Best Paper 项目页上的 Code / GitHub 链是开源核查主来源。
- **基线对照表** 比 PDF 更易扫：VTex、神经场、NvDiffRec* 与 HKTex 并排，适合写 wiki「结论」与 Sim 资产选型。
- 与 [`circle-group/hktex`](https://github.com/circle-group/hktex) README 的 `optimisation.py` 配置族互证。

## 关联资料

- 论文归档：[`sources/papers/hktex_eccv_2026_arxiv_2609_07557.md`](../papers/hktex_eccv_2026_arxiv_2609_07557.md)
- 代码仓库：[`sources/repos/hktex.md`](../repos/hktex.md)
