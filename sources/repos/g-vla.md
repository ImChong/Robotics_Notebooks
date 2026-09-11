# G-VLA（airvlab/G-VLA）

> 来源归档

- **标题：** GVLA — Gripper-aware Vision Language Action Models（项目页仓库）
- **类型：** repo（**静态项目页**，非训练实现）
- **组织：** AIRV Lab，利物浦大学（<https://github.com/airvlab>）
- **代码：** <https://github.com/airvlab/G-VLA>
- **项目页：** <https://airvlab.github.io/G-VLA/>
- **论文：** <https://arxiv.org/abs/2608.24603>
- **数据集：** <https://huggingface.co/datasets/GVLA/MiGA-Dataset>
- **入库日期：** 2026-09-11
- **一句话说明：** 托管 GVLA 官方项目页的 GitHub Pages 源仓；README 仅描述 MiGA 与 GVLA 论文贡献，**无训练/推理/评测脚本**。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [GVLA 论文实体](../../wiki/entities/paper-gvla-gripper-aware-vla.md) | **canonical 论文节点**；本仓为项目页镜像，非实现入口 |
| [MiGA 数据集](../datasets/miga-dataset.md) | 数据在 HF `GVLA/*`；LeRobot 格式 |
| [VLA](../../wiki/methods/vla.md) | gripper-aware conditioning + dual MoA 微调框架 |
| [LeRobot](../../wiki/entities/lerobot.md) | MiGA 子集按 LeRobot Parquet 发布 |

## 工程要点（README 摘要，2026-09-11）

- **内容：** 项目页 HTML/资源 + BibTeX；致谢 nerfies 模板。
- **缺失：** 无 `requirements.txt`、无 `train.py` / `eval.py`、无 checkpoint 发布说明。
- **复现读法：** 截至入库日仅可下载 **MiGA** 做数据实验；GVLA 训练需等待官方实现或自研复现论文 §4。

## 开源状态

- **部分开源边界：** 本仓库 = **已开源的项目页**；**GVLA 训练代码与权重 ≠ 本仓**（待官方后续发布）。
