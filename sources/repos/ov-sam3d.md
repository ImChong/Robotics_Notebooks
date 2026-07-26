# OV-SAM3D

> 来源归档

- **标题：** Open-Vocabulary SAM3D: Understand Any 3D Scene
- **类型：** repo
- **论文：** https://arxiv.org/abs/2405.15580
- **项目页：** https://hithqd.github.io/projects/OV-SAM3D/
- **代码：** https://github.com/HanchenTai/OV-SAM3D
- **入库日期：** 2026-07-26
- **一句话说明：** 训练无关的开放词汇 3D 场景理解：超点粗 mask ← 多视角 SAM 反投影修正，再结合 RAM 开放标签与重叠分数合并实例；偏 **离线点云理解**，非机载实时主路径。
- **沉淀到 wiki：** [GO2 三维语义建图 Query](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)

---

## README 要点

1. `generate_coarse_masks.py`：基于 superpoints + SAM 生成粗 3D mask。
2. `refine_masks.py`：开放标签与重叠分数细化。
3. 评测路径对齐 OpenMask3D 风格；在 ScanNet200 / nuScenes 等上报告。

## 开源状态

- **已开源**：`HanchenTai/OV-SAM3D`。
- **边界：** 适合离线着色/实例理解与伪标注；实时 GO2 机载需另选 DualMap / OVO / FindAnything 类系统。

## 对 wiki 的映射

- Query：[go2-3d-semantic-mapping-sam-pipeline](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md)
