# scan2mesh_cvpr_2019_dai

> 来源归档（ingest）

- **标题：** Scan2Mesh: From Unstructured Range Scans to 3D Meshes
- **短名：** Scan2Mesh（Dai & Nießner，CVPR 2019；**注意与 PUMA 等几何 scan-to-mesh 管线同名不同物**）
- **类型：** paper
- **来源：** CVPR 2019 Open Access / arXiv
- **原始链接：**
  - <https://arxiv.org/abs/1811.10464>
  - <https://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Scan2Mesh_From_Unstructured_Range_Scans_to_3D_Meshes_CVPR_2019_paper.pdf>
- **项目页：** <https://niessnerlab.org/projects/dai2019scan2mesh.html> — 归档见 [`sources/sites/niessnerlab-scan2mesh.md`](../sites/niessnerlab-scan2mesh.md)
- **作者：** Angela Dai, Matthias Nießner
- **机构：** 慕尼黑工业大学（Technical University of Munich, TUM）
- **出处：** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2019, pp. 5574–5583
- **入库日期：** 2026-09-15
- **一句话说明：** **数据驱动生成式** 方法：将 **非结构化、可能不完整** 的 range scan 直接预测为 **indexed face set**（顶点 + 面索引），通过 **顶点 / 边 / 面** 分阶段 proxy loss + CNN/GNN 实现离散一对一映射，输出更接近艺术家 CAD 的紧凑 mesh，而非隐式函数 iso-surface。

## 核心摘录

### 1) 问题与动机
- 传统重建多走 **隐式函数 / 体素 / 后处理 marching** → mesh 往往过密、噪声多，与 **手工 CAD** 结构差异大。
- 目标：从 **单次或稀疏 range scan** 直接生成 **结构化三角 mesh**（顶点 + 面索引），边 sharper、更 compact。

### 2) 方法要点
1. **分阶段生成：** 依次预测 **vertices → edges → faces**，每阶段用 **proxy loss** 与 GT 建立 **一对一离散对应**。
2. **网络：** **卷积网络** 处理 scan 特征 + **图神经网络** 处理 mesh 拓扑阶段。
3. **输出：** **indexed face set**（非点云后处理 mesh），条件于输入 scan。
4. **与 PUMA 区分：** PUMA 是 **几何 SLAM** 中的 scan-to-**已有地图 mesh** 配准；本文是 **学习式 scan→mesh 生成**，不涉及 LiDAR 里程计。

### 3) 实验（论文摘要）
| 项 | 内容 |
|----|------|
| 数据 | ShapeNet 等合成 / 扫描数据（论文以 object-level range scan 为主） |
| 对比 | 相对隐式函数重建：更 **sharp、compact** 的 artist-like mesh |
| 局限 | 物体级生成；非大规模室外 LiDAR SLAM 地图 |

### 4) 开源核查（步骤 2.5）
- **项目页（2026-09-15 核查）：** 提供 **Paper** 链接；**未列** 官方 GitHub / 代码下载。
- **第三方：** 存在非官方复现（如 `sirine90/Scan2Mesh` TUM 课程项目），**不可** 当作官方实现。
- **结论：** **确认未开源**（截至入库日项目页无官方代码链）→ wiki `## 源码运行时序图` 标不适用。

## 对 wiki 的映射

- 升格 [Scan2Mesh（CVPR 2019）论文实体](../../wiki/entities/paper-scan2mesh-cvpr2019-dai.md)
- 与 [PUMA](../../wiki/entities/paper-puma-lidar-mesh-odometry.md) 建立「同名不同物」交叉引用
- 可选链到 [Points as Tori](../../wiki/entities/paper-points-as-tori.md)（mesh/SDF 重建路线对照）

## 当前提炼状态

- [x] 摘要 + 分阶段生成 + 开源边界 + 与 PUMA 消歧
- [x] wiki 实体页
- [x] `sources/sites/`
