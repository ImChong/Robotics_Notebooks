# maker-rat.github.io/morph（X-Morph 项目页）

- **标题：** X-Morph — Human Motion Priors for Scalable Robot Learning Across Morphologies
- **类型：** site / project-page
- **URL：** <https://maker-rat.github.io/morph/>
- **配套论文：** [X-Morph（arXiv:2606.30290）](https://arxiv.org/abs/2606.30290) — 归档见 [`sources/papers/xmorph_arxiv_2606_30290.md`](../papers/xmorph_arxiv_2606_30290.md)
- **代码：** 截至 **2026-08-06** 项目页 **Code** 按钮为 `disabled`（无 GitHub URL）
- **入库日期：** 2026-08-06

## 一句话摘要

NUS **X-Morph** 官方站点：展示人体运动经跨形态重定向后驱动 **Go2 / Yuna hexapod / 带臂四足** 的 locomotion、物体交互与文本条件行为迁移；强调「人体运动可作非人形腿式机器人的可复用行为先验」。

## 公开信息要点（截至入库日）

- **机构 / 作者：** National University of Singapore；Ritwik Sharma*†、Shivam Sood*、Arhaan Jain、Shyam Charan Kesavamoorthi、Chengyang He、Guillaume Adrien Sartoretti。
- **页首入口：** Paper（PDF）/ arXiv 可用；**Code**、**Video** 按钮均为 disabled。
- **演示板块：**
  - Pipeline Overview / Live Interactive Demo（占位 *Coming soon*）
  - Locomotion skills：Walk / Turn / Squat
  - Object Interaction：挪障、推物、抬物；下游开门
  - Text-conditioned：Kimodo 文本→G1→X-Morph 重定向→同一跟踪栈
  - Generalization：Go2 右爪泛化、Yuna 宽前肢动作
- **BibTeX：** 页内仍为占位 `@article{xmorph2026,...}`（author/journal 空）。

## 为何值得保留

- **步骤 2.5 开源核查主入口：** 以项目页按钮状态判定代码是否已公开，避免仅凭 PDF 臆断。
- **非 PDF 证据：** 跨形态视频遥操作、文本条件与下游开门初始化比表格更直观。
- **与 arXiv 三角互证：** 摘要、平台与 Kimodo/GMR 部署分支一致。

## 关联资料

- 论文归档：[`sources/papers/xmorph_arxiv_2606_30290.md`](../papers/xmorph_arxiv_2606_30290.md)
- Wiki 实体：[wiki/entities/paper-xmorph.md](../../wiki/entities/paper-xmorph.md)
