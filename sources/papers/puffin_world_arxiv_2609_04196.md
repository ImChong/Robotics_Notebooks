# Puffin-World: Scaling a Unified Multimodal Model with Native 3D World States（arXiv:2609.04196）

> 来源归档（ingest）

- **标题：** Puffin-World: Scaling a Unified Multimodal Model with Native 3D World States
- **简称：** Puffin-World
- **类型：** paper / world-model / multimodal / 3d-generation
- **arXiv：** <https://arxiv.org/abs/2609.04196>
- **PDF：** <https://arxiv.org/pdf/2609.04196>
- **项目页：** <https://kangliao929.github.io/projects/puffin-world/> — 归档见 [`sources/sites/puffin-world-project.md`](../sites/puffin-world-project.md)
- **HF Blog：** <https://huggingface.co/blog/KangLiao/puffin-world>
- **代码：** <https://github.com/KangLiao929/Puffin>（`Puffin-World/` 子目录；**已开源**，NTU S-Lab License 1.0）
- **模型：** <https://huggingface.co/KangLiao/Puffin-World>（Base / Pro / Caption）
- **数据：** <https://huggingface.co/datasets/KangLiao/Puffin-16M>（Puffin-Cam-15M + Puffin-Traj-1M + 28 数据集相机标注）
- **机构：** 南洋理工大学 S-Lab（NTU）、密歇根大学、北京交通大学、大晓机器人（ACE Robotics）等
- **入库日期：** 2026-09-10
- **一句话说明：** 用 physics / geometry / appearance 三类原生 3D 世界状态统一多模态世界模型；Omni-Camera 9 通道条件 + LLM 理解 + 扩散生成，支持单图相机到世界理解、可控视角仿真、图文到 3D 世界与重建；Puffin-16M 规模化训练。

## 开源状态（步骤 2.5，2026-09-10）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线（交互 demo、评测表、Puffin-16M 说明） |
| GitHub | **已开源**（~472★）：训练 / 评测 / `scripts/demo/{world_modeling,physics_perception,spatial_simulation}.py` |
| 权重 | **已发布**（HF：`Puffin-World-Base/Pro/Caption.pth`，2026-08-23） |
| 数据集 | **已发布**（Puffin-16M + Bench；28 数据集 ~44.5M 图像相机标注集合） |

**结论：已开源** — 可复现 demo 与训练管线；许可为 NTU S-Lab License 1.0（非 Apache）。

## 核心摘录

### 摘录 1：三类原生世界状态

- **Physics：** 重力场与纬度图，锚定真实世界朝向与地平线。
- **Geometry：** 深度，暴露场景 3D 结构。
- **Appearance：** RGB 图像/序列，与相机、几何条件联合生成而非孤立像素预测。

**对 wiki 的映射：** [paper-puffin-world](../../wiki/entities/paper-puffin-world.md)

### 摘录 2：Omni-Camera 与物理传播

- 每像素 9 通道条件 = 3 通道绝对视角场（up vector + latitude）+ 6 通道相对射线（origin + direction）。
- 单图估计 roll / pitch / vFoV；长轨迹上通过相对旋转 \(R^{rel}_{t\leftarrow 0}\) 传播参考重力 \(g_0\) 到各目标视角。

**对 wiki 的映射：** [paper-puffin-world](../../wiki/entities/paper-puffin-world.md)

### 摘录 3：Puffin-16M 与评测

- Puffin-Cam-15M（15M VL-camera 三元组）+ Puffin-Traj-1M（1M 轨迹，含 360° 探索）。
- 相机理解：Stanford2D3D / MegaDepth / TartanAir / LaMAR 上 median error 与 AUC 领先。
- 3D 世界：RealEstate10K PSNR **17.22**、LPIPS **0.318**；Puffin-Traj-Bench median roll **0.80°**、pitch **1.10°**。

**对 wiki 的映射：** [paper-puffin-world](../../wiki/entities/paper-puffin-world.md)、[generative-world-models](../../wiki/methods/generative-world-models.md)

## 当前提炼状态

- [x] 项目页 + GitHub + HF 核查（2026-09-10）
- [x] wiki 映射：`wiki/entities/paper-puffin-world.md`
