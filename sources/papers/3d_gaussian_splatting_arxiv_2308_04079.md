# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

> 来源归档（foundational paper + official project page + authors' implementation）

- **标题：** 3D Gaussian Splatting for Real-Time Radiance Field Rendering
- **作者：** Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
- **机构：** Inria、Université Côte d’Azur、Max Planck Institute for Informatics（MPI Informatik）
- **发表：** ACM Transactions on Graphics 42(4), SIGGRAPH 2023
- **arXiv：** <https://arxiv.org/abs/2308.04079>
- **官方项目页：** <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>
- **作者代码：** <https://github.com/graphdeco-inria/gaussian-splatting>
- **入库日期：** 2026-10-22
- **一句话说明：** 以可优化的各向异性 3D 高斯显式表示场景，从稀疏校准点初始化并交替优化高斯与密度控制，再以可见性感知的 splatting 渲染，实现高视觉质量的实时新视角合成。
- **沉淀到 wiki：** [具身感知六种空间表征](../../wiki/concepts/embodied-perception-six-spatial-representations.md)

## 核心摘录

### 表示与优化

- 每个 3D Gaussian 具有位置、协方差（尺度与旋转）、不透明度及球谐外观属性；场景从相机校准得到的稀疏点云初始化。
- 训练过程中交错执行参数优化与密度控制（克隆、分裂、剪枝），以细化场景表示；快速可见性感知光栅化支持训练与实时渲染。
- 论文报告在 1080p 下实现至少 30 FPS 的新视角合成，并在多个数据集上展示视觉质量与速度的折衷。该指标是论文场景与实现结果，不代表任意机器人硬件上的端到端吞吐。

### 开源状态与复现入口

项目页提供论文与作者实现入口，`graphdeco-inria/gaussian-splatting` 是作者公开的参考代码仓库。论文设定侧重从已校准多视图重建静态场景；动态场景、物理参数与机器人可碰撞几何需要额外建模。

## 对 wiki 的映射

- [具身感知六种空间表征](../../wiki/concepts/embodied-perception-six-spatial-representations.md) — 将 3DGS 与 NeRF 作为场景外观/几何表示，与导航所需地图语义区分。
- [3D Gaussian Splatting 论文索引](../../wiki/entities/paper-rcl-2308-04079-3d-gaussian-splatting-for-real-time-radiance-fie.md) — 现有 WAM 策展索引节点；本归档补充原始来源与方法摘录。
- [GS-Playground](../../wiki/entities/gs-playground.md) — 将批量 3DGS 渲染接入并行物理仿真。

## 参考来源

- [arXiv 摘要与论文](https://arxiv.org/abs/2308.04079)
- [官方项目页](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [作者代码仓库](https://github.com/graphdeco-inria/gaussian-splatting)
