# Points as Tori: Fast Pointwise Signed Distance for Point Clouds（arXiv:2607.16946）

> 来源归档（ingest）

- **标题：** Points as Tori: Fast Pointwise Signed Distance for Point Clouds
- **类型：** paper / geometry processing / signed distance / point cloud reconstruction
- **arXiv：** <https://arxiv.org/abs/2607.16946>（PDF：<https://arxiv.org/pdf/2607.16946.pdf>；作者页 PDF：<https://nzfeng.github.io/research/PointsAsTori/PointsAsTori.pdf>）
- **项目页：** <https://nzfeng.github.io/research/PointsAsTori/index.html>
- **代码：** <https://github.com/nzfeng/points-as-tori>
- **作者：** Nicole Feng；Ioannis Gkioulekas†；Keenan Crane†（† 同等贡献）
- **机构：** Carnegie Mellon University（卡内基梅隆大学）
- **发表：** ACM Transactions on Graphics（SIGGRAPH 2026）；DOI [10.1145/3811385](https://doi.org/10.1145/3811385)
- **入库日期：** 2026-09-12
- **一句话说明：** **Points as Tori（PAT）** 用局部环面闭式 SDF + 预训练网络拟合，从带法向点云直接输出可任意点查询的解析 SDF；无需体素离散或全局优化，并给出卷积距离公式与 winding number / Poisson 重建的统一理论。

## 开源状态（项目页 + 仓库核查，2026-09-12）

- **已开源、可运行推理：** 项目页链到 [`nzfeng/points-as-tori`](https://github.com/nzfeng/points-as-tori)。`pip install .` 构建扩展；`PointsAsTori` API 含预训练 `FundamentalFormPredictor.pkl`；`demo/demo_3d.py` 可视化；`training/train.py` + Google Drive 6.2 GB 训练数据。
- **许可：** MIT。
- **边界：** PyPI 发布为 TODO；demo 依赖 pyglet/imgui 较重；环面预计算随点云规模线性增长（百万点级需数十秒–数分钟）。

## 摘要级要点

- **问题：** 点云 SDF 往往要么快（启发式/卷积距离）但不鲁棒，要么鲁棒（全局重建/优化）但慢且需离散化。
- **PAT 思路：** 每点拟合环面（二阶局部面 + 闭式 SDF）；查询时用自归一化指数加权平均各点环面距离（Eq. 1）。
- **学习用量极小：** 固定大小邻域 → 小网络输出曲率/偏移；**一次预训练**后泛化到任意点云（非 per-shape 神经隐式）。
- **理论贡献：** 卷积距离公式统一 LogSumExp、winding number、Poisson 重建等；证明 naive 卷积距离无法从采样几何泛化；自归一化变体 + 可控 \(g_i\) 才可恢复真实 SDF 景观。
- **应用：** offset、布尔/形态学、sphere tracing；输入含摄影测量、mesh 采样、3D Gaussian、神经隐式。

## 核心论文摘录（MVP）

### 1) 自归一化卷积距离 + 环面局部 SDF

- **链接：** 项目页 Eq. (1)–(3)；论文 §3
- **摘录要点：** \(\phi(\mathbf{x})=\frac{\sum_i g_i(\mathbf{x})\exp(-\lambda_{\mathbf{x}}\|\mathbf{x}-\mathbf{p}_i\|)}{\sum_i \exp(-\lambda_{\mathbf{x}}\|\mathbf{x}-\mathbf{p}_i\|)}\)；\(g_i\) 为拟合环面的 SDF。相对 Eq. (2) 单点卷积公式，自归一化允许通过 \(g\) 控制最近点处流形景观。
- **对 wiki 的映射：**
  - [Points as Tori](../../wiki/entities/paper-points-as-tori.md) — 核心机制与流程图。
  - [Collision Distance Optimization](../../wiki/concepts/collision-distance-optimization.md) — SDF 查询在规划中的位置。

### 2) 数据驱动环面拟合 vs 端到端神经 SDF

- **链接：** 项目页「Why learning?」「Why not end-to-end learning?」；论文 §4
- **摘录要点：** 经典核带宽/邻域启发式在边/角处脆弱；用小网络在合成邻域上预训练（≈45 h / RTX 3090，6.4 MB）。避开 per-shape 拟合 eikonal 的局部极小与昂贵推理。
- **对 wiki 的映射：**
  - [Points as Tori](../../wiki/entities/paper-points-as-tori.md) — 工程实践与预计算成本表。
  - [Embodied Perception Six Spatial Representations](../../wiki/concepts/embodied-perception-six-spatial-representations.md) — 点云 vs 体素 ESDF 选型。

### 3) 应用、局限与 Signed Heat Method 对照

- **链接：** 项目页 Limitations；论文 §6
- **摘录要点：** 支持百万–千万点但预计算线性增长；OOD 几何可能弱于非学习全局法（如 signed heat method）；未来方向含更快预计算、可微优化、压缩表示。
- **对 wiki 的映射：**
  - [Points as Tori](../../wiki/entities/paper-points-as-tori.md) — 局限与机器人部署读法。
  - [Smooth Navigation Path Generation](../../wiki/methods/smooth-navigation-path-generation.md) — 导航 SDF 软惩罚接口。

## BibTeX

```bibtex
@article{Feng:2026:PAT,
  author = {Feng, Nicole and Gkioulekas, Ioannis and Crane, Keenan},
  title = {Points as Tori: Fast Pointwise Signed Distance for Point Clouds},
  journal = {ACM Trans. Graph.},
  volume = {45},
  number = {4},
  articleno = {53},
  year = {2026},
  month = jul,
  doi = {10.1145/3811385},
  url = {https://doi.org/10.1145/3811385}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-points-as-tori.md`](../../wiki/entities/paper-points-as-tori.md)
- 代码归档：[`sources/repos/points-as-tori.md`](../repos/points-as-tori.md)
- 项目页：[`sources/sites/points-as-tori.md`](../sites/points-as-tori.md)
- 互链：[Collision Distance Optimization](../../wiki/concepts/collision-distance-optimization.md)、[Embodied Perception Six Spatial Representations](../../wiki/concepts/embodied-perception-six-spatial-representations.md)、[Grasp Pose Estimation](../../wiki/methods/grasp-pose-estimation.md)
