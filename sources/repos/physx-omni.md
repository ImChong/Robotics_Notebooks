# physx-omni / PhysX-Omni

> 来源归档

- **标题：** PhysX-Omni（官方实现：生成、评测与数据工具）
- **类型：** repo
- **维护者：** physx-omni（S-Lab NTU 等）
- **链接：** <https://github.com/physx-omni/PhysX-Omni>
- **项目页：** <https://physx-omni.github.io/>
- **许可证：** S-Lab License（见仓库 `LICENSE`）
- **入库日期：** 2026-05-30
- **一句话说明：** 开源 **PhysX-Omni** 训练/推理（Qwen2.5-VL 微调 + 几何解码 + URDF/XML 导出）、**PhysX-Bench** 评测脚本，以及 **PhysXVerse** 预处理与多数据集训练配置入口。
- **沉淀到 wiki：** 是 → [`wiki/entities/physx-omni.md`](../../wiki/entities/physx-omni.md)

## 仓库结构要点（README，2026-05-30）

| 模块 | 说明 |
|------|------|
| `qwen-vl-finetune/` | 在 PhysXNet / PhysX-Mobility / PhysXVerse 上微调 VLM；`scripts/train_physx.sh` |
| 推理脚本 | `1vlm_demo.py`（VLM）→ `2infer_geo.py`（解码）→ `3jsongen_update.py`（URDF & XML） |
| `benchmark/` | **PhysX-Bench** 资产生成、VLM 评测、分母校验与聚合（见 `benchmark/README.md`） |
| `dataset/` | PhysXVerse 体素化与 RLE 表征预处理（`1voxel_verse.py` 等） |
| `applications_scene/` | 基于既有工作的简易 **仿真就绪场景** 生成 |
| `convert_objects2scene.py` | 多物体合并为 sim-ready 场景 |

## 依赖与生态（Acknowledgement）

- 数据与代码 lineage：**PartNet-mobility**、**Qwen**、**TRELLIS**、**Depth-Anything**、**Grounded-SAM**、**CAST**；训练数据另需 [PhysXNet](https://huggingface.co/datasets/Caoza/PhysX-3D)、[PhysX-Mobility](https://huggingface.co/datasets/Caoza/PhysX-Mobility)、[PhysXVerse](https://huggingface.co/datasets/PhysX-Omni/PhysXVerse)。
- 环境：`setup.sh`（TRELLIS 风格）或 `requirements.txt` + conda `physx-omni`；推理权重 `python download.py` 从 HF 拉取。

## 对 wiki 的映射

- **实体页**：[`wiki/entities/physx-omni.md`](../../wiki/entities/physx-omni.md)
- **论文摘录**：[`sources/papers/physx_omni_arxiv_2605_21572.md`](../papers/physx_omni_arxiv_2605_21572.md)
- **项目页**：[`sources/sites/physx-omni-github-io.md`](../sites/physx-omni-github-io.md)
