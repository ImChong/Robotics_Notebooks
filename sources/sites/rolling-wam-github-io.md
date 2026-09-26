# rolling-wam.github.io（Rolling-WAM 项目页）

- **标题：** Rolling-WAM: World Action Models with Rolling Imagination
- **类型：** site / project-page
- **URL：** <https://rolling-wam.github.io/>
- **配套论文：** [Rolling-WAM（arXiv:2609.30247）](https://arxiv.org/abs/2609.30247)
- **代码：** <https://github.com/zyinghua/Rolling-WAM>（**待发布**，见 [`sources/repos/rolling-wam.md`](../repos/rolling-wam.md)）
- **入库日期：** 2026-09-26

## 一句话摘要

USC 等提出的 **滚动想象 WAM**：在 replan 循环间 **分摊 video–action 联合去噪**，steady-state **215 ms**、相对 Joint-WAM **4.5×**；LIBERO / RoboTwin 2.0 / **G1 真机** 报告 competitive SR。

## 公开信息要点（截至入库日）

- **Method：** Refine window → Execute & observe → Roll window；MoT + attention mask 示意图。
- **数字：** 4.5×、98.1%、93.3%、85.0% G1；replanning latency 条形图（Joint 978 / Fast 548 / Rolling 215 ms）。
- **Real-world：** 与 Joint-WAM、Fast-WAM 并列 qualitative（部分视频 placeholder）。

## 关联资料

- 论文归档：[`sources/papers/rolling_wam_arxiv_2609_30247.md`](../papers/rolling_wam_arxiv_2609_30247.md)
- 仓库占位：[`sources/repos/rolling-wam.md`](../repos/rolling-wam.md)
