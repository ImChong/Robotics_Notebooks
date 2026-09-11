# hktex_eccv_2026_arxiv_2609_07557

> 来源归档（ingest）

- **标题：** Heat Kernel Textures: the Geodesic Gaussians That Do Not Splat
- **短名：** HKTex
- **类型：** paper
- **来源：** arXiv abs / PDF；ECCV 2026 正式出版
- **原始链接：**
  - <https://arxiv.org/abs/2609.07557>
  - <https://arxiv.org/pdf/2609.07557>
  - <https://doi.org/10.1007/978-3-032-37595-7_17>
- **项目页：** <https://circle-group.github.io/research/HeatKernelTextures/> — 归档见 [`sources/sites/circle-group-heat-kernel-textures.md`](../sites/circle-group-heat-kernel-textures.md)
- **代码：** <https://github.com/circle-group/hktex> — 归档见 [`sources/repos/hktex.md`](../repos/hktex.md)
- **作者：** Simone Foti<sup>*</sup>, Caner Korkmaz<sup>*</sup>, Stefanos Zafeiriou, Tolga Birdal（<sup>*</sup> equal contribution）
- **机构：** Imperial College London（帝国理工学院）
- **版本：** arXiv:2609.07557（2026-09）；ECCV 2026 **Best Paper** + **Long Oral**
- **入库日期：** 2026-09-11
- **一句话说明：** 在三角网格上用各向异性热核（测地线高斯）做无 UV 内在纹理；黎曼梯度下降 + 曲面自适应 densify/prune，接 Mitsuba 可微物理渲染；可从现有 UV 纹理或多视角图像拟合，存储约为 UV 纹理 **1/10**，感知质量优于 **5–10×** 存储的神经基线。

## 核心摘录

### 1) 问题
- **UV 纹理长期痛点：** 展开接缝、面积扭曲、图集浪费、重复顶点、texel 分辨率随曲率不均——在机器人 Real2Sim 资产、游戏与 PBR 管线里都是隐性成本。
- **3D Gaussian Splatting 启发但不同：** 3DGS 用欧氏空间各向异性高斯做新视角合成；HKTex 把「高斯」类比搬到 **离散黎曼曲面** 上的 **测地线热核**，**不做 splat**，而是内在定义在 mesh 上。
- **神经纹理基线：** 顶点色、MLP positional encoding、Instant-NGP 式内在场、VTex 等可在 mesh 上表达外观，但常需更大存储或难与 PBR 材质分解对齐。

### 2) 方法要点
1. **表示：** 每个热核在曲面上由源点位置、扩散角、各向异性（方向与拉伸）、尺度、锐度与 RGB 调制；核在测地邻域内扩散，叠加成内在纹理场。
2. **优化：** 核参数与位置在曲面上用 **黎曼梯度下降 + 动量**（依赖 [`digeo`](https://github.com/circle-group/DiGeo) 等离散几何工具）；位置更新强制留在三角 mesh 上。
3. **密度控制：** 曲面感知的 **重要性剪枝** + **误差驱动 densify**——低贡献核删除，欠重建区域沿流形主方向 **clone / split** 核（类比 3DGS adaptive control，但操作在 2D 流形上）。
4. **渲染：** 与 **Mitsuba** 可微光线追踪集成，支持 **albedo + 多材质通道** 的 PBR 分解渲染。
5. **拟合模式：**
   - **现有 UV 纹理：** 在任意曲面点评估 HKTex 并与 UV 采样 GT 对齐；
   - **多视角逆渲染：** 多相机 RGB（及材质监督）经光线追踪渲染后与观测比对反传。
6. **加速：** KNN 加速热核评估（`texture_hktex_knn.yaml` / `multiview_hktex_knn_ray_small.yaml`）。

### 3) 实验（论文 / 项目页报告摘要）
- **存储：** 相对标准 UV 纹理，HKTex 约 **1/10** 内存（用户摘要与项目页「considerably lowering memory footprint」一致）。
- **质量：** 在 UV 纹理拟合与多视角逆渲染设定下，感知质量 **优于使用 5–10× 存储** 的神经基线（MLP pos. encoding、Instant-NGP 式内在场、VTex、NvDiffRec* 等；见项目页交互对比）。
- **任务轴：** (a) 从 GT UV 图压缩重拟合 + 材质旋转展示；(b) 直接从多视角图像优化外观与材质分解。
- **荣誉：** ECCV 2026 **Best Paper Award** + **Long Oral**。

### 4) 局限
- 依赖 **三角 mesh** 与（多视角模式下）相机标定；不是点云或隐式场通用表示。
- 热核数量与 KNN 邻域仍影响速度与显存；超高分细节需足够 densify。
- 官方环境钉定 **Python 3.11 + CUDA 12.9 + Mitsuba 3.7**；神经基线另需 `hktex-mlp` 环境与 tiny-cuda-nn。
- 论文主线是 **外观 / 纹理**，不直接解决动力学或碰撞网格生成。

### 5) 开源核查（步骤 2.5）
- **项目页（2026-09-11）：** 链到 arXiv、DOI 与 GitHub [`circle-group/hktex`](https://github.com/circle-group/hktex)；交互 demo 展示与 VTex / MLP / NvDiffRec* 等对比。
- **仓库：** MIT License；主入口 `optimisation.py` + `configs/`；`hktex/` 含 data、trainers、rendering、knn_heat、density_controllers；`scripts/` 含 benchmark 与论文图复现。
- **结论：** **已开源、可运行** 纹理拟合与多视角 Mitsuba 渲染实验。wiki 须写 `## 源码运行时序图`。

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-hktex-heat-kernel-textures.md`](../../wiki/entities/paper-hktex-heat-kernel-textures.md)
- 交叉 [`wiki/entities/paper-lego-leveled-language-gaussian-splatting.md`](../../wiki/entities/paper-lego-leveled-language-gaussian-splatting.md)（3DGS 表亲但任务不同）、[`wiki/entities/paper-simfoundry-real2sim-scene-generation.md`](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md)（sim 资产纹理）、[`wiki/entities/gs-playground.md`](../../wiki/entities/gs-playground.md)（3DGS 渲染用于 RL）

## 当前提炼状态

- [x] 摘要 + 方法主干 + 存储/质量要点 + 开源边界
- [x] wiki 实体页、仓库与项目页归档
- [ ] 若后续接入 Objaverse 批量 benchmark 脚本，可补评测表数字到实体页
