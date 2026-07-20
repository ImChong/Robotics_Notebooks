# yucrazing.github.io/clothtransformer（ClothTransformer 项目页）

- **标题：** ClothTransformer — Unified Latent-Space Transformers for Scalable Cloth Simulation
- **类型：** site / project-page
- **URL：** <https://yucrazing.github.io/clothtransformer/>
- **配套论文：** [ClothTransformer（arXiv:2605.27852）](https://arxiv.org/abs/2605.27852) — 归档见 [`sources/papers/clothtransformer_arxiv_2605_27852.md`](../papers/clothtransformer_arxiv_2605_27852.md)
- **代码：** <https://github.com/YuCrazing/ClothTransformer> — 归档见 [`sources/repos/YuCrazing-ClothTransformer.md`](../repos/YuCrazing-ClothTransformer.md)
- **数据集：** <https://huggingface.co/datasets/YuCrazing1/ClothTransformer-dataset>
- **入库日期：** 2026-07-20

## 一句话摘要

NTU S-Lab 等团队的 **ClothTransformer** 官方站点：展示 **单模型** 在 **人体着装、机器人抓取布料、布料–刚体碰撞** 三类场景上的泛化视频；强调 **latent-space Transformer**、**~493.4k 帧无穿透数据集** 与 **可微 CCD** 消融。

## 公开信息要点（截至入库日）

- **机构：** S-Lab, Nanyang Technological University；Feeling AI；University of Oxford；Shanghai AI Laboratory（与 arXiv 作者单位一致）。
- **页首卖点：** *One Model, Diverse Scenarios* — 无需 per-scenario fine-tuning。
- **演示板块：**
  - **Diverse Object Collision** — 布料落于未见刚体（剑、角色等）
  - **Human Garment** — 未见 body/garment/animation 组合（前空翻、舞蹈等）
  - **Robotic Manipulation** — 未见布料网格被夹爪抓取抬起
  - **Unified Architecture** — Spatial Encoder / Temporal Transformer / Spatial Decoder 总览图
  - **Comparison** — 相对 prior SOTA 的误差与视觉对比
  - **Penetration-Free Dataset** — 三场景 ~493.4k frames 展示
  - **Differentiable CCD** — w/ DCD → +CCD Loss → +CCD Post. 渐进消融
- **BibTeX：** `@misc{zhang2026clothtransformerunifiedlatentspacetransformers,...}`

## 开源核查（步骤 2.5）

| 资源 | 状态 | 链接 |
|------|------|------|
| **数据集** | **已发布**（2026-07-17） | Hugging Face `YuCrazing1/ClothTransformer-dataset` |
| **代码** | **部分 / 占位** | GitHub 仓仅 README + `.gitignore`，无训练推理脚本 |
| **预训练权重** | **截至入库日未列** | 项目页与 README 均未给出 checkpoint URL |

## 为何值得保留

- **非 PDF 证据：** 三场景并排视频比表格更直观呈现 **统一模型泛化** 与 **CCD 穿透抑制**。
- **与 arXiv / HF 三角互证：** 数据集发布日期与 README News 一致；架构图与论文 Figure 2 对齐。
- **机器人相关入口：** Robotic Manipulation 子集连接 **可变形体操作仿真** 与 manipulation 任务线。

## 关联资料

- 论文归档：[`sources/papers/clothtransformer_arxiv_2605_27852.md`](../papers/clothtransformer_arxiv_2605_27852.md)
- 代码仓库：[`sources/repos/YuCrazing-ClothTransformer.md`](../repos/YuCrazing-ClothTransformer.md)
- Wiki 实体：[`wiki/entities/paper-clothtransformer-unified-latent-cloth-simulation.md`](../../wiki/entities/paper-clothtransformer-unified-latent-cloth-simulation.md)
