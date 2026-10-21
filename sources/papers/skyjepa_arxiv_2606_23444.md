# SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors（arXiv:2606.23444）

> 来源归档（ingest）

- **标题：** SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors
- **类型：** paper / JEPA / quadrotor / sim-to-real control
- **arXiv：** <https://arxiv.org/abs/2606.23444>
- **代码 / 项目页：** arXiv 页未提供；页面状态为 Under Review
- **作者（arXiv 页）：** Pratyaksh Rao；Wancong Zhang；Randall Balestriero；Yann LeCun；Giuseppe Loianno
- **机构：** arXiv 页未显示；文章称为伯克利、纽约大学、布朗大学联合发布（**来自文章转述，未在 arXiv 页核对**）
- **提交：** 2026-06-22（v1），2026-06-23（v2），cs.RO / cs.LG
- **入库日期：** 2026-10-20
- **来源文章：** [wechat_deepblue_lecun_jepa_11_papers_2026-10-20.md](../blogs/wechat_deepblue_lecun_jepa_11_papers_2026-10-20.md)

## 开源核查（步骤 2.5，2026-10-20）

arXiv 页仅有 PDF / HTML / TeX，无代码或项目页 → **未发现官方代码**。

## 摘录要点（归纳）

- 潜动力学模型 + 物理启发探针（把冻结潜变量映射为可解释状态）+ 嵌入式可运行的采样型最优控制。
- 训练数据来自仿真流水线（摘要）；文章转述 500 套仿真域、2 万条轨迹，室外真实试飞未经实机微调。
- 文章转述消融：60 步推演位置误差由 5.56 m 降至 1.43 m（替换物理探针后）。

## 对 wiki 的映射

- 主节点：[paper-skyjepa](../../wiki/entities/paper-skyjepa.md)
