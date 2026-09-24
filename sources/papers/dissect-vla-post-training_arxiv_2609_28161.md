# Dissecting Advantage-Guided Post-Training for Vision-Language-Action Policies

> 来源：[具身智能小站 · 13 篇盘点](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)（2026-09-24）

## 元数据

- **arXiv：** [2609.28161](https://arxiv.org/abs/2609.28161)
- **PDF：** https://arxiv.org/pdf/2609.28161
- **项目页：** https://dissectvla.github.io/
- **开源结论（2026-09-24）：** **待发布**

## 核心摘录

- **一句话：** 把 VLA 优势引导后训练拆成 **构造 / 校准 / 利用** 三阶段，用离线诊断筛设计再少跑真机。
- **机制：** Stage I：IQL + n-step TD 优势；Stage II：Value-based 分组校准（低 η²）；Stage III：连续 advantage weight（非 filter）。
- **指标：** 四双臂真机任务：mean task progress **+0.42**、success **+0.63** vs SFT init；Weight 条件 mean success **0.74**。

## 对 wiki 的映射

- 实体页：[DissectVLA](../../wiki/entities/paper-dissect-vla-post-training.md)
