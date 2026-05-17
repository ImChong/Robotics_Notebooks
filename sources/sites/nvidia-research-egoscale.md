# NVIDIA Research — EgoScale（GEAR Lab）

> 来源归档（ingest）

- **标题：** EgoScale
- **类型：** site（官方项目页）
- **发布方：** NVIDIA Corporation / NVIDIA GEAR
- **原始链接：** <https://research.nvidia.com/labs/gear/egoscale/>
- **页面标注日期：** Feb 19, 2026（以项目页为准）
- **入库日期：** 2026-05-17
- **一句话说明：** 官方对外页：强调 **>20k 小时** egocentric 人视频预训练 **流式 VLA**、**人数据规模–验证损失** 的 log-linear 规律及其与 **真机灵巧操作** 的关联，并展示 **人预训练 + 人–机对齐 mid-training + 任务后训练** 的管线与 **五类高灵巧** 评测任务演示。

## 摘录要点（与论文分工）

- **管线叙述：** 先用腕运动 + 重定向灵巧手动作在海量人视频上预训练，再用 **对齐的人–机 play** 做轻量 mid-training，最后任务后训练；页面以交互轮播概括 **22k h 预训练** 与 **对齐 play** 数据属性。
- **架构一句：** **VLM 骨干 + DiT 动作专家** 的 flow-based VLA；人/机通过 **共同腕级动作表示** 统一，**本体与手部** 经 **per-embodiment 轻量适配器** 处理（与论文 Figure 2(b) 叙述一致）。
- **任务展示：** Shirt Rolling、Tongs、Card Sorting、Bottle、Syringe 等真机 rollout（页面含视频占位与文字说明）。
- **代码状态：** 项目页写明 **GitHub（Coming Soon）** —— 入库时无公开代码仓链接；后续若发布应单独 `sources/repos/` 索引并回链本页。

## BibTeX

页面提供的 BibTeX 与 [论文摘录](../papers/egoscale_arxiv_2602_16710.md) 一致（`@misc` / `eprint=2602.16710`）。

## 对 wiki 的映射

- [EgoScale](../../wiki/methods/egoscale.md) — 面向读者的入口摘要、演示链接与「工程状态（代码待发布）」注记
