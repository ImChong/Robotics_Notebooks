# Matterport3D: Learning from RGB-D Data in Indoor Environments（arXiv:1709.06158）

> 来源归档（原文通读 · 支撑 `wiki/entities/matterport3d-simulator.md`）

- **标题：** Matterport3D: Learning from RGB-D Data in Indoor Environments
- **作者：** Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Nießner, Manolis Savva, Shuran Song, Andy Zeng, Yinda Zhang（Princeton University / TUM / Stanford University）
- **类型：** paper / dataset / 3d-reconstruction / rgbd（3DV 2017）
- **原文：** <https://arxiv.org/abs/1709.06158>（2026-11 通过 arXiv PDF 获取全文）
- **项目页：** <https://niessner.github.io/Matterport/>
- **仓库：** <https://github.com/niessner/Matterport>
- **一句话说明：** 构建大规模真实建筑尺度室内 RGB-D 数据集 Matterport3D（90 栋建筑、10,800 个全景视点、194,400 张 HDR 1280×1024 RGB-D 图像），提供全局配准、纹理网格与全景语义标注，是 Matterport3D Simulator 与 R2R 等 VLN 基准的场景资产基石。
- **核对范围：** 通读 arXiv PDF §1–§6（引言、相关工作、§3 数据采集与标注、§4 数据集统计与 5 项视觉基准任务、§5 实验结果、§6 总结）；核心仓库目录通过 `niessner/Matterport` 核对（`code/gaps`、`tasks/` 5 项视觉任务脚本）与 `peteanderson80/Matterport3DSimulator` 核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **数据规模（§3 & Table 1）：** 涵盖 **90 栋**多类型真实建筑（主要住宅与商业建筑，按建筑划分 61 train / 11 val / 18 test），共 **10,800 个全景（panoramas）**、**194,400 张** HDR 1280×1024 RGB-D 图像（每个全景包含 3 个俯仰角 × 6 个方位角 = **18 组视角**）。
- **空间与几何覆盖（§3 & §4）：** 共 2,056 个已标注房间、**46,561 m²** 地面面积（平均每栋 517 m²、22.8 个房间），全景采集点平均间距约 **2.25 m**；全局 bundle adjustment 后跨扫描局部对齐误差典型值 **< 1 cm**，重建出高精度纹理表面网格。
- **语义与实例标注（§3.3）：** 众包标注房间边界/类别与 **50,811 个物体实例**，归纳映射到 **40 个规范物体类别（canonical categories）**，提供 2D 像素级语义、3D 体素语义及实例边框。
- **5 项原始视觉任务（§5 & `tasks/`）：** 关键点匹配（keypoint matching）、视点重叠估计（view overlap prediction）、表面法向估计（surface normal estimation）、区域分类（region classification）与 3D 语义体素标注（semantic voxel labeling），展示全景上下文（多视视野）比单视角显著提升下游性能。
- **数据获取边界（`niessner/Matterport`）：** 场景纹理图像、Mesh、深度与全景数据受 Matterport 学术使用许可协议约束，需申请下载脚本而非随 Git 仓库直接分发。

## 对 wiki 的映射

- [matterport3d-simulator](../../wiki/entities/matterport3d-simulator.md)
- [paper-vln-01-r2r](../../wiki/entities/paper-vln-01-r2r.md)
- [vision-language-navigation](../../wiki/tasks/vision-language-navigation.md)

## 参考来源

- 论文：<https://arxiv.org/abs/1709.06158>
- 仓库：<https://github.com/niessner/Matterport>
- 仿真器仓库：<https://github.com/peteanderson80/Matterport3DSimulator>
