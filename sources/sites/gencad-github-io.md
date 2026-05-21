# GenCAD 项目主页（gencad.github.io）

> 来源归档

- **标题：** GenCAD: Image-conditioned Computer-Aided Design Generation with Transformer-based Contrastive Representation and Diffusion Priors
- **类型：** site（项目页 / 交互 Demo / BibTeX）
- **链接：** https://gencad.github.io/
- **入库日期：** 2026-05-21
- **一句话说明：** MIT **Decode** 组展示的 **图像条件 CAD program 生成** 项目站：强调输出为 **完整参数化命令历史** 而非仅 mesh；提供 **生成**、**同图多样性**、**top-3 检索** 等交互演示与 arXiv:2409.16294 BibTeX。
- **论文：** arXiv:2409.16294
- **代码：** https://github.com/ferdous-alam/GenCAD
- **沉淀到 wiki：** 是 → [`wiki/entities/gencad.md`](../../wiki/entities/gencad.md)

## 项目页公开主张（归纳）

- **核心卖点：** GenCAD 生成 **3D CAD 实体** 的同时输出 **CAD program**，保留工程可编辑性；对比 mesh/voxel/point cloud 路线在 **精度与可修改性** 上的损失。
- **架构四步（页面文案）：** Transformer 潜表示 → 对比学习联合 CAD–图像潜空间 → 图像条件潜扩散 → 解码为命令序列。
- **演示区块：** Image-conditional CAD generation；Sample diversity；Image-conditional CAD retrieval（约 7000 程序库 top-3）。

## 对 wiki 的映射

- **实体页**：[`wiki/entities/gencad.md`](../../wiki/entities/gencad.md) — 与 [`sources/papers/gencad_arxiv_2409_16294.md`](../papers/gencad_arxiv_2409_16294.md)、[`sources/repos/ferdous-alam-gencad.md`](../repos/ferdous-alam-gencad.md) 交叉校验。
