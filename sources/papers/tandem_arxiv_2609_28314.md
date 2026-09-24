# TANDEM: Task and Motion Planning with As-Needed Demonstrations for Efficient Vision-Language-Action Model Fine-tuning

> 来源：[具身智能小站 · 13 篇盘点](../../sources/blogs/wechat_embodied_13_papers_forgetmimic_2026-09-24.md)（2026-09-24）

## 元数据

- **arXiv：** [2609.28314](https://arxiv.org/abs/2609.28314)
- **PDF：** https://arxiv.org/pdf/2609.28314
- **项目页：** https://prpl-group.com/tandem/
- **开源结论（2026-09-24）：** **待发布**

## 核心摘录

- **一句话：** 规划器能走的步骤交给 TAMP，只在能力缺口处按需遥操作，并把各阶段拼成完整 VLA 微调示范。
- **机制：** VLM 扩展 TAMP 域：发明缺失 predicate + 人类执行的 magic operator；每段人工后重感知并验证 effect 再续规划；DATAFARM 对齐 TAMP 段与 VLA 预训练分布。
- **指标：** 代表任务同等人工时间下示范量约为全程遥操作 **2.9×**；五任务各 **20** 条示范微调 π0.5-DROID，平均成功率 **0%→60%**。

## 对 wiki 的映射

- 实体页：[TANDEM](../../wiki/entities/paper-tandem.md)
