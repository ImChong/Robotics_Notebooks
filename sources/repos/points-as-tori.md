# nzfeng/points-as-tori

> 来源归档

- **标题：** Points as Tori（PAT）官方实现
- **类型：** repo
- **组织 / 作者：** Nicole Feng、Ioannis Gkioulekas、Keenan Crane（CMU）
- **代码：** <https://github.com/nzfeng/points-as-tori>
- **项目页：** <https://nzfeng.github.io/research/PointsAsTori/index.html>
- **论文：** <https://arxiv.org/abs/2607.16946>
- **许可：** MIT（`LICENSE`）
- **入库日期：** 2026-09-12
- **一句话说明：** 从带法向点云估计 SDF 的 C++/Python 实现：`pip install .` 构建 nanobind 扩展；`PointsAsTori` 类一次预计算环面参数后支持 `signed_distance` / `sdf_gradient` 点查询；含预训练网络权重、`training/` 训练脚本与 `demo/demo_3d.py` 可视化。

## 入口速查（对齐 README / `infer.py`）

| 路径 / 命令 | 作用 |
|-------------|------|
| `git submodule update --init --recursive` | 拉取 nanobind、nanoflann、libigl、fcpw 等子模块 |
| `pip install .` | 构建 C++ 扩展并安装 `pointsastori` Python 包 |
| `src/pointsastori/infer.py` | **`PointsAsTori(points, normals)`** API：`signed_distance(queries)`、`sdf_gradient(queries)` |
| `src/pointsastori/models/FundamentalFormPredictor.pkl` | 预训练环面拟合网络（≈6.4 MB） |
| `src/pointsastori/network.py` | 网络结构定义 |
| `src/cpp/torus_distance.cpp` | 环面闭式 SDF 与梯度（OpenMP 并行） |
| `demo/demo_3d.py` | pyglet + pyimgui 切面 sphere-tracing 可视化（Python 3.9–3.11） |
| `training/train.py` | 训练环面拟合网络；依赖见 `training/` README |
| Google Drive 训练数据 | <https://drive.google.com/drive/folders/1hnG-1OCwZ0SWS47Kgmk4chPG825AOfGt>（6.2 GB） |

## 性能口径（README，RTX 3090）

| 点云规模 | 环面预计算 | 单次查询 |
|----------|------------|----------|
| 100k | <10 s | 10⁻⁴–10⁻³ s |
| 1M | <60 s | 同上 |
| 5–10M | 1–4 min | 同上 |
| 20M+ | >10 min | 同上 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Points as Tori](../../wiki/entities/paper-points-as-tori.md) | 论文实体：PAT 方法、理论统一、局限 |
| [Collision Distance Optimization](../../wiki/concepts/collision-distance-optimization.md) | 机器人规划中的 SDF / 距离查询子问题 |
| [Embodied Perception Six Spatial Representations](../../wiki/concepts/embodied-perception-six-spatial-representations.md) | 点云 / TSDF / ESDF 表示选型 |
| [Smooth Navigation Path Generation](../../wiki/methods/smooth-navigation-path-generation.md) | 导航优化中的障碍 SDF 软惩罚 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/points_as_tori_arxiv_2607_16946.md`](../papers/points_as_tori_arxiv_2607_16946.md)
- 项目页：[`sources/sites/points-as-tori.md`](../sites/points-as-tori.md)
- 沉淀 **[`wiki/entities/paper-points-as-tori.md`](../../wiki/entities/paper-points-as-tori.md)**
