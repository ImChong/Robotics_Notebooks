# DiffGI: Differentiable Geometry Images for High-Fidelity Thin-Shell 3D Generation

> 来源归档（ingest）

- **标题：** DiffGI: Differentiable Geometry Images for High-Fidelity Thin-Shell 3D Generation
- **类型：** paper
- **来源：** arXiv abs / HTML；项目页交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2607.13365>
  - <https://ar5iv.labs.arxiv.org/html/2607.13365>
  - <https://ejshim.github.io/diffgi/>
  - <https://github.com/EJShim/diffgi>（截至入库日仅项目页静态资源，无可运行训练/推理入口）
- **作者：** Eungjune Shim, Hansol Lee, Eunjung Ju
- **机构：** CLO Virtual Fashion Inc.（韩国）
- **venue / 状态：** ECCV 2026（项目页 BibTeX `shim2026diffgi`）
- **入库日期：** 2026-07-27
- **一句话说明：** 用 **连续 2D TSDF geometry image + Differentiable Marching Squares** 把薄壳/非流形表面（服装等）压入 **32×32×4** 潜空间，再以 DiT/UNet 潜扩散做标签 / 单视图 / 缝纫图案条件生成；相对 TRELLIS / GarmageNet 在 GarmageSet 上边界更锐、网格更紧（约 **23K** 顶点），消费级 GPU ~**1.2 s**、CPU ~**8.5 s**。

## 开源核查（2026-07-27）

| 项 | 状态 |
|----|------|
| 项目页 | <https://ejshim.github.io/diffgi/> — Paper 链到 arXiv；**Code (soon)** 按钮禁用 |
| GitHub | <https://github.com/EJShim/diffgi> — 公开，但树内仅 `docs/`（项目页镜像）+ `.gitignore`，**无** train/eval/权重 |
| 数据集 | 评测用 ABO / GarmageSet / WARDROBE（外部已有数据集）；本工作未另发自有权重包 |
| 结论 | **宣称将开源 / 待发布** — 截至 **2026-07-27** 无可运行官方实现；勿按已开源复现选型 |

## 核心论文摘录（MVP）

### 1) 问题：体素隐式难表薄壳；旧 geometry image 用二值占用

- **链接：** <https://arxiv.org/abs/2607.13365> §I–II
- **摘录要点：** SDF/occupancy/NeRF 类生成默认 **封闭体**，服装等薄壳易人工加厚或前后粘连，且 Marching Cubes 网格常缺 UV，难接仿真/材质管线。Omages / GIMDiffusion / GarmageNet 走 multi-chart geometry image，但仍依赖 **二值 occupancy**，边界随分辨率台阶化，重建后处理不可微。
- **对 wiki 的映射：**
  - [DiffGI（论文实体）](../../wiki/entities/paper-diffgi.md) — 问题定位。
  - [PhysForge](../../wiki/entities/paper-physforge-physics-grounded-3d-assets.md) — 体素/隐式资产生成对照。
  - [ClothTransformer](../../wiki/entities/paper-clothtransformer-unified-latent-cloth-simulation.md) — 下游布料仿真对 UV-friendly 薄壳网格的需求。

### 2) 表示：连续 2D TSDF + Differentiable Marching Squares

- **链接：** arXiv §III.1–III.2；项目页 Pipeline
- **摘录要点：**
  - Offline Mesh→DiffGI：UV packing → \(1024^2\) position map → dilation → 像素域 TSDF（截断 15 px）→ 双线性下采样至 \(256\times256\times4\)。
  - **DMS：** 对 TSDF 零交叉用线性插值求子像素 UV 顶点，再经 position map 双线性采样得 3D；拓扑查表离散，顶点坐标对 \(\phi\) 可微（含 \(\epsilon\) 防爆炸）；鞍点 Case 6/9 固定拆成独立片。
  - 复杂度 \(O(N^2)\)，对比 3D 可微等值面 \(O(N^3)\)。
- **对 wiki 的映射：**
  - [DiffGI](../../wiki/entities/paper-diffgi.md) — 核心原理与流程。
  - [Differentiable Simulation](../../wiki/concepts/differentiable-simulation.md) — 「可微」语境对照（本文是 **2D 等值面提取**，非接触动力学）。

### 3) DiffGI-VAE + 潜扩散条件生成

- **链接：** arXiv §III.3–III.4
- **摘录要点：**
  - VAE 以 SD1.5 权重初始化，压到 \(32\times32\times4\)；\(\mathcal{L}=\mathcal{L}_{Pos}+\lambda_{TSDF}\mathcal{L}_{TSDF}+\lambda_{Normal}\mathcal{L}_{Normal}+\lambda_{KL}\mathcal{L}_{KL}\)，法向图经 **nvdiffrast** 对 DMS 网格可微渲染。
  - 条件生成：标签 → DiT-B/2；单视图 → DINOv2-L + DiT-L/2 cross-attn；缝纫图案/占用 → UNet-Tiny；训练期 UV chart 90° 旋转重打包增强。
- **对 wiki 的映射：**
  - [DiffGI](../../wiki/entities/paper-diffgi.md) — 方法栈。
  - [Articraft](../../wiki/entities/articraft.md) — 另一条「可接下游管线」的 3D 资产生成路线。

### 4) 评测：重建、消融、效率、image-to-3D

- **链接：** arXiv §IV；Table 1–5；项目页 Results
- **摘录要点：**
  - **VAE 重建（Table 1）：** GarmageSet 上 DiffGI CD \(0.46\times10^{-3}\) vs Omages \(1.31\) / GarmageNet Official \(1.89\)；潜空间 \(32\times32\times4\) 仍优于更大/未压缩表示。
  - **消融（Table 2）：** TSDF ≫ Occ；再加法向损失 NC \(0.921\to0.961\)。
  - **效率（Table 3）：** Image 条件 RTX 4070 **3.22 GB / 1.21 s**；MacBook M4 CPU **8.52 s**；TRELLIS-image 同卡 **16.28 GB / 4.52 s**。
  - **Image-to-3D（Table 5）：** DiffGI 平均 **23K** 顶点，CD **1.35**、F1 **0.48**、BCD **2.91**，优于 TRELLIS / TRELLIS.2 / GarmageNet。
- **对 wiki 的映射：**
  - [DiffGI](../../wiki/entities/paper-diffgi.md) — 评测表与结论。

### 5) 局限

- **链接：** arXiv §V
- **摘录要点：** TSDF 线性插值在极尖机械棱上可能圆角；**不生成 RGB/PBR**；各 UV chart 独立重建可在 chart 接缝处开缝，影响下游物理仿真；未来拟 dual contouring、纹理/材质联合生成、图案布局与 3D 形状两阶段解耦。
- **对 wiki 的映射：**
  - [DiffGI](../../wiki/entities/paper-diffgi.md) — 局限与开源边界。
  - [Sim2Real](../../wiki/concepts/sim2real.md) — 资产几何一致性提醒。
