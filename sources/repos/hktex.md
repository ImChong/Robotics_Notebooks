# circle-group/hktex

> 来源归档

- **标题：** Heat Kernel Textures: the Geodesic Gaussians That Do Not Splat
- **类型：** repo
- **组织 / 作者：** circle-group（Simone Foti / Caner Korkmaz / Stefanos Zafeiriou / Tolga Birdal，Imperial College London）
- **代码：** <https://github.com/circle-group/hktex>
- **项目页：** <https://circle-group.github.io/research/HeatKernelTextures/>
- **论文：** arXiv:2609.07557 — [`sources/papers/hktex_eccv_2026_arxiv_2609_07557.md`](../papers/hktex_eccv_2026_arxiv_2609_07557.md)
- **许可：** MIT License
- **入库日期：** 2026-09-11
- **一句话说明：** 官方 HKTex 实现：各向异性热核纹理 + 黎曼优化 + 曲面 densify/prune + Mitsuba 可微渲染；`optimisation.py` 驱动 UV 纹理拟合与多视角逆渲染实验。
- **沉淀到 wiki：** [`wiki/entities/paper-hktex-heat-kernel-textures.md`](../../wiki/entities/paper-hktex-heat-kernel-textures.md)

## 开源核查（2026-09-11）

| 项 | 状态 |
|----|------|
| 仓库可见 | 是（公开；MIT） |
| 语言 / 栈 | Python 3.11；PyTorch 2.10+cu129；torch_geometric；digeo；Mitsuba 3.7 |
| 可运行入口 | **有** — `python optimisation.py --config configs/...` |
| 权重 | 无预训练场景纹理包；用户提供 `data.mesh_path` 或多视角观测 |
| License | MIT |
| 结论 | **已开源、可运行** 训练 / 渲染 / benchmark 脚本 |

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `optimisation.py` | 主实验入口；`--config` + dot 覆盖（如 `data.mesh_path=...`） |
| `configs/texture_hktex_knn.yaml` | 从带 UV 纹理的 mesh 拟合 KNN 加速 HKTex |
| `configs/multiview_hktex_knn_ray_small.yaml` | 多视角观测 + Mitsuba 光线追踪拟合 HKTex |
| `configs/texture_mlp.yaml` / `multiview_mlp_ray.yaml` | 神经纹理基线（需 `hktex-mlp` 环境 + tiny-cuda-nn） |
| `configs/multiview_vertex_ray.yaml` | 顶点色基线 |
| `configs/ablations/` | 消融配置 |
| `hktex/data/` | Mesh 与观测数据模块 |
| `hktex/modules/` | 纹理、几何与插值模型 |
| `hktex/knn_heat/` | KNN 热核实现 |
| `hktex/density_controllers/` | 自适应密度控制（剪枝 / densify） |
| `hktex/rendering/` | 可微与 Mitsuba 渲染器 |
| `hktex/trainers/` | 优化工作流 |
| `scripts/` | Benchmark、计时、可视化与论文图 |
| `interactive_*.py` | Jupyter 交互入口 |
| `outputs/` | 默认日志、渲染与 checkpoint 目录 |

环境：`mamba create -n hktex python=3.11.13` + README 依赖列表；神经基线另建 `hktex-mlp`。目标平台：**Linux + NVIDIA GPU（CUDA 12.9 驱动）**。

## 对 wiki 的映射

- 论文：[`sources/papers/hktex_eccv_2026_arxiv_2609_07557.md`](../papers/hktex_eccv_2026_arxiv_2609_07557.md)
- 项目页：[`sources/sites/circle-group-heat-kernel-textures.md`](../sites/circle-group-heat-kernel-textures.md)
- 沉淀 **[`wiki/entities/paper-hktex-heat-kernel-textures.md`](../../wiki/entities/paper-hktex-heat-kernel-textures.md)**
