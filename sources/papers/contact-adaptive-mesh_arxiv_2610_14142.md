# Refine Where It Matters: Contact-Adaptive Meshing for Deformable Object Manipulation

> 来源归档（paper）

- **标题：** Refine Where It Matters: Contact-Adaptive Meshing for Deformable Object Manipulation
- **中文标题：** 按需细化：面向可变形物体操作的接触自适应网格划分
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.14142>
- **入库日期：** 2026-10-21
- **线索来源：** [公众号综述（33 篇 Manipulation）](../raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)（具身运控lab，2026-10-20）
- **一句话说明：** 仿真循环内按接触与形变的分布细化四面体网格，用 GPU 并行选择无冲突的细化边。
- **沉淀到 wiki：** [`wiki/entities/paper-contact-adaptive-meshing.md`](../../wiki/entities/paper-contact-adaptive-meshing.md)

## 摘要要点（arXiv 摘要页）

- 均匀精细网格冗余，粗网格可能欠解析接触形变。
- 方法：复用求解器已有量打分候选边，GPU 并行选择无冲突子集并原地更新拓扑与状态。

## 开源状态

- **未核查代码**：arXiv 摘要页未列代码链接。
