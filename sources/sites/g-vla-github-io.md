# GVLA Project Page（airvlab.github.io/G-VLA）

> 来源归档（ingest）

- **标题：** GVLA: Gripper-aware Vision Language Action Models（项目主页）
- **类型：** project site
- **官方入口：** <https://airvlab.github.io/G-VLA/>
- **论文：** <https://arxiv.org/abs/2608.24603>
- **项目 PDF：** <https://airvlab.github.io/G-VLA/GVLA-Gripper-aware-Vision-Language-Action-Models.pdf>
- **GitHub（项目页源）：** <https://github.com/airvlab/G-VLA>
- **数据集：** <https://huggingface.co/datasets/GVLA/MiGA-Dataset>
- **HF org：** <https://huggingface.co/GVLA>
- **机构：** 利物浦大学 AIRV Lab 等（见论文作者单位）
- **入库日期：** 2026-09-11
- **一句话说明：** ECCV 2026 录用页：MiGA 103K / 5 gripper / 36 tasks；GVLA multi-gripper tokenizer + dual MoA；仿真与 UR5 真机结果。

## 开源状态（项目页核查，2026-09-11）

| 项 | 状态 |
|----|------|
| Paper | arXiv **2608.24603** / ECCV 2026 |
| MiGA 数据 | HF **`GVLA/*`** 子集已发布（LeRobot Parquet；Apache-2.0） |
| GVLA 代码/权重 | 项目页 **未列** 训练仓库或 checkpoint 下载；GitHub 仓仅为静态站 |
| Lab | <https://github.com/airvlab> |
| 结论 | **部分开源** — 数据可用；**模型与训练栈待发布** |

## 页面结构（策展）

| 区块 | 内容要点 |
|------|----------|
| Banner | **Accepted at ECCV 2026** |
| MiGA | 103K trajectories；5 gripper types；36 tasks；Real + Sim |
| GVLA Method | Multi-gripper tokenization；Dual Mixture-of-Adapters |
| Results | 仿真四任务类别 + 零样本/少样本 + UR5 Robotiq 2F-85 真机 |
| BibTeX | `@inproceedings{zhang2026gvla,...}` |

## 对 wiki 的映射

- 论文：[`sources/papers/gvla_arxiv_2608_24603.md`](../papers/gvla_arxiv_2608_24603.md)
- 数据：[`sources/datasets/miga-dataset.md`](../datasets/miga-dataset.md)
- 沉淀 **[`wiki/entities/paper-gvla-gripper-aware-vla.md`](../../wiki/entities/paper-gvla-gripper-aware-vla.md)**
