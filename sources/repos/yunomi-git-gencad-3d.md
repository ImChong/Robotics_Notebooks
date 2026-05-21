# yunomi-git / GenCAD-3D

> 来源归档

- **标题：** GenCAD-3D（官方实现）
- **类型：** repo
- **维护者：** Nomi Yu（yunomi-git）
- **链接：** https://github.com/yunomi-git/GenCAD-3D
- **项目页：** https://gencad3d.github.io/
- **论文：** arXiv:2509.15246
- **数据：** https://huggingface.co/datasets/yu-nomi/GenCAD_3D
- **权重：** https://huggingface.co/yu-nomi/GenCAD_3D
- **入库日期：** 2026-05-21
- **一句话说明：** **多模态几何→CAD program** 全栈：自编码器、对比学习、条件扩散的训练/评测脚本；**SynthBal** 合成数据生成；**Onshape** 导出；依赖 **pythonocc-core** 与 PyTorch（README 示例 torch 2.7.1）。
- **沉淀到 wiki：** 是 → [`wiki/entities/gencad-3d.md`](../../wiki/entities/gencad-3d.md)

## 仓库公开结构（README 摘要）

| 子模块 | 作用 |
|--------|------|
| `autoencoder/` | CAD 序列自编码；`synthbal/` 增广数据集生成与合并 |
| `contrastive/` | 点云 / mesh FeaStNet 对比训练与检索评测 |
| `diffusion/` | 条件潜扩散训练、生成/重建指标与可视化 |
| `GenCADGenerator/` | `program_to_cad`：h5 程序 → Onshape Part Studio |

## 数据集分包（Hugging Face，须解压至 `DATA_PATH`）

- **GenCAD3D**（约 127 GB）：CAD、点云、网格、STL、STEP
- **GenCAD3D_Scans**（约 700 MB）：真实扫描 + 净扫 + CAD + 几何
- **GenCAD3D_SynthBal**（约 109 GB）：SynthBal 增广
- **GenCAD3D_SynthBal_1M**（约 9 GB）：仅 CAD program 的百万级子集

## 快速推理（README）

```bash
python -m diffusion.evaluation.visualize_diffusion_inference \
  -encoder_type mesh_feast \
  -contrastive_model_name mesh_SynthBal_1M_SBD \
  -filenames examples_files/00152170.stl
```

## 对 wiki 的映射

- **实体页**：[`wiki/entities/gencad-3d.md`](../../wiki/entities/gencad-3d.md)
