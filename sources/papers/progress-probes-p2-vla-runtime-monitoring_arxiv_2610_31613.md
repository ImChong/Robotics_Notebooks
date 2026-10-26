# Progress Probes P²（arXiv:2610.31613）

> 来源归档（paper）

- **标题：** Progress Probes P²: Lightweight Runtime Monitoring for Vision-Language-Action Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.31613>
- **PDF：** <https://arxiv.org/pdf/2610.31613>
- **作者：** Atiksh Bhardwaj, Edward W. Duan, Junwon Seo, Prithwish Dan, Wei-Chiu Ma, Andrea Bajscy, Preston Culbertson
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-10-26
- **一句话说明：** 进度探针 P²：面向视觉语言动作模型的轻量化运行时监测。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
2. **周报所述问题：** VLA 执行过程中需要判断任务是否正常推进，但持续调用大型视觉语言模型进行状态评估会增加运行开销。
3. **周报所述创新点：** 引入轻量级进度监测模块，在策略运行时评估任务执行状态，为动作继续执行、重新规划和失败检测提供反馈。
4. **arXiv 摘要（英文节选）：** Vision-Language-Action (VLA) models can perform diverse manipulation tasks directly from visual observations and natural language instructions, but they remain prone to mistakes and unpredictable failures. Deploying these policies without continuous human supervision will require robust runtime monitors that detect when the policy has gone off-course. Existing monitors use external foundation models, such as vision-language models (VLMs) to supervise the policy, introducing significant complexity and computational cost for monitoring at runtime.

**对 wiki 的映射**

- [paper-progress-probes-p2-vla-runtime-monitoring](../../wiki/entities/paper-progress-probes-p2-vla-runtime-monitoring.md)
