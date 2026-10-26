# Echo-VLA（arXiv:2610.31569）

> 来源归档（paper）

- **标题：** Echo: Coupled Action and Task Status Signal Denoising for Vision Language Action Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.31569>
- **PDF：** <https://arxiv.org/pdf/2610.31569>
- **作者：** Yufeng Liu, Bohao Fan, Juntu Zhao, Xinyi Wang, Hang Yu, Wenzhuan Luo, Yingzhi Wang, Qinghai Li, Yingdong Hu, Junliang Guo, Richard Wang, Dequan Wang, Yang Gao, Cheng Ma
- **arXiv 评论栏：** Project page: this https URL
- **外链：** <https://lyfeng001.github.io/Echo/>
- **入库日期：** 2026-10-26
- **一句话说明：** Echo：面向视觉语言动作模型的动作与任务状态信号联合去噪。

## 开源状态

- arXiv 摘要页/评论给出项目页或外链（https://lyfeng001.github.io/Echo/）；未发现代码仓库链接（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】Manipulation 37篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
2. **周报所述问题：** VLA 需要同时决定下一步动作和判断当前任务状态。如果动作生成与任务状态估计分别学习，两者可能出现不一致。
3. **周报所述创新点：** Echo 将动作与任务状态信号放入耦合去噪框架，使动作生成过程能够利用任务状态信息，并建立动作预测与任务进度估计之间的联系。
4. **arXiv 摘要（英文节选）：** Recognizing task completion and failure enables robots to stop upon success and initiate recovery or replanning after failure. Existing methods augment Vision Language Action (VLA) policies with task status signal prediction to support these execution decisions. However, learning and integrating these signals can require fine-grained supervision, external monitoring models, or complex architectural designs.

**对 wiki 的映射**

- [paper-echo-coupled-action-task-status-denoising](../../wiki/entities/paper-echo-coupled-action-task-status-denoising.md)
