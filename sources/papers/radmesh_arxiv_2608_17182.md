# RADmesh: Remesh-Aware Mesh Deformation

> 来源归档（ingest）

- **标题：** RADmesh: Remesh-Aware Mesh Deformation
- **类型：** paper
- **来源：** arXiv abs / HTML；项目页与 GitHub 交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2608.17182>
  - <https://threedle.github.io/radmesh/>
  - <https://github.com/threedle/radmesh>（**已开源**，ECCV 2026 Oral 官方实现）
- **作者：** Nam Anh Dinh, Itai Lang, Oded Stein, Rana Hanocka
- **机构：** University of Chicago（3D 视觉 / Threedle Lab）；University of Southern California；Technion
- **venue / 状态：** **ECCV 2026（Oral）**
- **入库日期：** 2026-08-20
- **一句话说明：** 在 **显式三角网格** 上直接做 **文本引导生成式形变**：用 **per-vertex 6D 形变量 Q**（方向 + 局部缩放，扩展 Geometry in Style 的 dARAP）配合 **周期性 Botsch–Kobbelt 各向同性 remesh** 与 **Adam 状态插值**，在 **CSD（DeepFloyd IF）** 视觉监督下实现 **局部大尺度生长** 与 **全局 detailization**；粗到细目标边长日程对质量至关重要。

## 开源核查（2026-08-20）

| 项 | 状态 |
|----|------|
| 项目页 | <https://threedle.github.io/radmesh/> — Paper / Code / BibTeX 齐全；Code 链到 GitHub |
| GitHub | <https://github.com/threedle/radmesh> — **已开源**；含 `run_optimization.py`、`radmesh/deformations.py`、`env_setup.sh`、示例配置与 `example-run/` |
| 权重 | 依赖 Hugging Face **DeepFloyd/IF-I-XL-v1.0** 与 **IF-II-L-v1.0**（需接受许可并 `huggingface_hub login`） |
| GPU | README：单卡 **A40 / L40S（48GB）** 级；需 CUDA 与 `nvdiffrast` 运行时编译 |
| 结论 | **已开源、可运行** — 最短路径 `python run_optimization.py -c example-config-localized.json` |

## 核心论文摘录（MVP）

### 1) 问题：固定连通性的生成式网格形变难做大尺度结构变化

- **链接：** <https://arxiv.org/abs/2608.17182> §I–II
- **摘录要点：** TextDeformer / MeshUp / Geometry in Style 等 **视觉监督网格形变** 多在 **固定 triangulation** 下优化顶点位移或 per-face Jacobian / per-vertex rotation；大尺度 **生长、 appendage、薄片结构** 会过度拉伸单元。inverse rendering 侧 remesh（Palfinger、Barda 等）以 **顶点位置** 为变量，与生成式形变偏好的 **微分形变量** 不兼容。
- **对 wiki 的映射：**
  - [RADmesh（论文实体）](../../wiki/entities/paper-radmesh.md) — 问题定位与 remesh-in-the-loop 动机
  - [DiffGI](../../wiki/entities/paper-diffgi.md) — 同属 ECCV 2026 显式网格生成/编辑，但 DiffGI 走 2D TSDF geometry image + 潜扩散

### 2) 方法：Q 形变量 + 周期 remesh + 优化器状态插值

- **链接：** 项目页 Overview / Deformation method / Remeshing method
- **摘录要点：**
  - **形变量 Q**：每顶点 6 维（`q_dir` + `q_scale`）；LocalStep 求 rotation R_k 与 diagonal scale S_k，GlobalStep 为 dARAP 全局解。
  - **Remesh**：每 **N=100** epoch 对最新形变结果做 **2 次** Botsch–Kobbelt 迭代；**重投影 barycentric** 插值 **Adam 状态** 与 Q 到新网格。
  - **粗到细 target edge length**：局部生长从 1.7×→1.0× 平均边长；全局 deform 从 1.4×→1.0×；**一次性细 remesh 失败**（论文 Fig. 10）。
  - **监督**：nvdiffrast 可微渲染 + **Cascaded Score Distillation（CSD）** + DeepFloyd IF。
- **对 wiki 的映射：**
  - [RADmesh](../../wiki/entities/paper-radmesh.md) — 流程 Mermaid 与源码运行时序
  - [Differentiable Simulation](../../wiki/concepts/differentiable-simulation.md) — 「可微」语境对照（本文是 **网格形变 + 视觉 SDS**，非接触动力学）

### 3) 应用：局部生长、全局 detailization、UV/纹理工作流

- **链接：** 项目页 Gallery / Applications
- **摘录要点：** 可选 **顶点选择 mask** 仅改局部；选区外 **保对应、保 UV/纹理**；支持多 prompt 迭代加部件（项目页 workflow 图）。预处理含 bbox 归一化、局部 **法向 inflation** 启发式与 **首次 remesh**。
- **对 wiki 的映射：**
  - [RADmesh](../../wiki/entities/paper-radmesh.md) — 工程实践与局限
  - [PhysForge](../../wiki/entities/paper-physforge-physics-grounded-3d-assets.md) / [Articraft](../../wiki/entities/articraft.md) — sim-ready / 可关节资产管线对照（RADmesh 偏 **交互式网格编辑**）

### 4) 实验读法：视觉质量 + 三角形效率 + 消融

- **链接：** arXiv §4；项目页 ablation 叙事
- **摘录要点：** 定性展示长尾巴、翅膀、服装等 **远距生长** 与 extremity 细节；强调 **各向同性三角** 与 **自适应 triangulation**（球体 global deform 例）。消融：**scale 分量**、**optimizer 状态插值**、**coarse-to-fine remesh**、**初始 inflation** 均关键。
- **对 wiki 的映射：**
  - [RADmesh](../../wiki/entities/paper-radmesh.md) — 结论与选型

## 引用（BibTeX）

```bibtex
@inproceedings{dinh2026radmesh,
  title     = {RADmesh: Remesh-Aware Mesh Deformation},
  author    = {Dinh, Nam Anh and Lang, Itai and Stein, Oded and Hanocka, Rana},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## 对 wiki 的映射

- [paper-radmesh](../../wiki/entities/paper-radmesh.md)
