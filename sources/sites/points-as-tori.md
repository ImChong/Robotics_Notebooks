# nzfeng.github.io/research/PointsAsTori（Points as Tori 项目页）

- **标题：** Points as Tori: Fast Pointwise Signed Distance for Point Clouds
- **类型：** site / project-page
- **URL：** <https://nzfeng.github.io/research/PointsAsTori/index.html>
- **配套论文：** [arXiv:2607.16946](https://arxiv.org/abs/2607.16946) — PDF：<https://nzfeng.github.io/research/PointsAsTori/PointsAsTori.pdf>；归档见 [`sources/papers/points_as_tori_arxiv_2607_16946.md`](../papers/points_as_tori_arxiv_2607_16946.md)
- **代码：** <https://github.com/nzfeng/points-as-tori> — 归档见 [`sources/repos/points-as-tori.md`](../repos/points-as-tori.md)
- **入库日期：** 2026-09-12

## 一句话摘要

CMU 的 **Points as Tori（PAT）** 官方项目页：从带法向的点云直接估计 **有符号距离（SDF）**，用局部环面（torus）闭式 SDF + 预训练小网络拟合曲率/偏移，支持任意分辨率点查询，无需体素离散或全局优化；统一 winding number、Poisson 重建与卷积距离公式视角。

## 公开信息要点（截至入库日）

- **机构：** Carnegie Mellon University（Nicole Feng；Ioannis Gkioulekas†、Keenan Crane† 同等贡献）。
- **发表：** ACM Transactions on Graphics（**SIGGRAPH 2026**）；DOI [10.1145/3811385](https://doi.org/10.1145/3811385)。
- **核心公式：** 自归一化卷积距离——对查询点 \(\mathbf{x}\) 用指数加权平均各点环面 SDF \(g_i(\mathbf{x})\)；\(\lambda_{\mathbf{x}}\) 自动选取。
- **学习边界：** 仅对 **局部邻域环面拟合** 用一次预训练网络（≈6.4 MB，RTX 3090 预训练 ≈45 h）；推理侧不做 per-shape 端到端神经场拟合。
- **应用展示：** 点云 offset、形态学/布尔运算、sphere tracing 直接可视化 offset 面；输入可来自摄影测量、mesh、3D Gaussian、神经隐式。
- **步骤 2.5（开源核查）：** 页首 **GitHub repository** 明确列出「Python API, demo visualization, training scripts and data」→ **已开源**（MIT，见仓库 `LICENSE`）。预训练权重随仓 `FundamentalFormPredictor.pkl`；训练数据 Google Drive 6.2 GB 外链。

## 为何值得保留

- **机器人相关接口：** 点云 SDF 是避障 TrajOpt、穿透检测、场景重建与 Sim 资产导入的共性需求；PAT 强调 **点级并行 + 无体素网格**，适合把激光/RGB-D 点云直接接入距离查询管线。
- **理论页价值：** 项目页长文解释卷积距离公式为何对采样几何失效、自归一化变体 + 环面局部基为何可泛化，比 PDF 摘要更适合写 wiki「为什么重要」。
- **与 arXiv / GitHub 三角互证：** API（`infer.py`）、demo（`demo/demo_3d.py`）、`training/train.py` 均可在官方仓定位。

## 关联资料

- 论文归档：[`sources/papers/points_as_tori_arxiv_2607_16946.md`](../papers/points_as_tori_arxiv_2607_16946.md)
- 代码仓库：[`sources/repos/points-as-tori.md`](../repos/points-as-tori.md)
- 升格：[`wiki/entities/paper-points-as-tori.md`](../../wiki/entities/paper-points-as-tori.md)
