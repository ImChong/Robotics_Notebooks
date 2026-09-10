# Puffin Series（KangLiao929/Puffin）

> 来源归档

- **标题：** Puffin Series — Towards Unified Multimodal 3D World Models
- **类型：** repo
- **链接：** https://github.com/KangLiao929/Puffin
- **Stars：** ~472（2026-09-10）
- **许可：** NTU S-Lab License 1.0
- **入库日期：** 2026-09-10
- **一句话说明：** Puffin 系列统一多模态 3D 世界模型仓：含 ICLR 2026 **Puffin**（相机中心理解/生成）与 **Puffin-World**（原生 3D 世界状态 + 训练/评测/demo）。
- **代码：** https://github.com/KangLiao929/Puffin（**已开源**）
- **沉淀到 wiki：** [paper-puffin-world](../../wiki/entities/paper-puffin-world.md)
- **交叉归档：** [puffin_world_arxiv_2609_04196.md](../papers/puffin_world_arxiv_2609_04196.md)、[puffin-world-project.md](../sites/puffin-world-project.md)

---

## 子项目索引

| 子目录 | 论文 | 说明 |
|--------|------|------|
| [`Puffin/`](https://github.com/KangLiao929/Puffin/tree/main/Puffin) | arXiv:2510.08673（ICLR 2026） | *Thinking with Camera*；Puffin-4M |
| [`Puffin-World/`](https://github.com/KangLiao929/Puffin/tree/main/Puffin-World) | arXiv:2609.04196 | 原生 physics/geometry/appearance；Puffin-16M |

## Puffin-World 运行时入口（README，2026-08）

环境：`conda` + Python 3.10 + PyTorch 2.7.0 + CUDA 12.6；`pip install -r requirements.txt` + `flash-attn==2.8.3`。

| 脚本 | 用途 |
|------|------|
| `scripts/demo/world_modeling.py` | 给定初始视角 + 相机轨迹 → 多视角 RGB/深度 + 点云重建 |
| `scripts/demo/physics_perception.py` | 单图重力感知相机理解（roll/pitch/vFoV） |
| `scripts/demo/spatial_simulation.py` | 相机可控 text-to-image 空间仿真 |

权重：`huggingface-cli download KangLiao/Puffin-World --local-dir checkpoints`（Base / Pro / Caption）。

训练：`configs/pipelines/final_stage_*` 多阶段对齐与世界建模；评测见 `documents/EVALUATION.md`。

---

## 对 wiki 的映射

- [paper-puffin-world](../../wiki/entities/paper-puffin-world.md)
- [Generative World Models](../../wiki/methods/generative-world-models.md)
