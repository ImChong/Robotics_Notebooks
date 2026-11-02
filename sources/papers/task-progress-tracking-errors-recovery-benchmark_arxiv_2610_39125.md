# 任务进度追踪基准（arXiv:2610.39125）

> 来源归档（paper）

- **标题：** Tracking Task Progress Through Errors and Recovery: A Benchmark and Recurrent Distributional Model
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.39125>
- **PDF：** <https://arxiv.org/pdf/2610.39125>
- **作者：** Wang, Chen, Niu, Xinze, Zhu, Zihao, Zhou, Keru, Xu, Yicheng, Jin, Shihao, Wang, Wenhao, Jiang, Bowen et al.
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 通过错误与恢复跟踪任务进度：评测基准与循环分布模型。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 长程机器人操作中的任务进度可能倒退或恢复，单纯根据当前图像估计进度，难以区分视觉相似但执行阶段不同的状态。
3. **周报所述创新点：** 提出 ReSTEP 数据集，包含 3,156 条操作轨迹和逐帧错误标注。同时提出 REMORA，通过循环视觉记忆、分布式进度预测和因果峰值选择，持续跟踪任务进度，并识别执行错误与恢复过程。
4. **arXiv 摘要（英文节选）：** Operational strategies for long-horizon robotic tasks require reliable evaluation to determine whether execution is progressing, stalling, regressing, or recovering from errors. Yet real- world robot datasets that capture error recovery and provide frame-level error annotations remain scarce. To address this gap, we introduce and open-source ReSTEP, a dataset of 3,156 episodes across two manipulation tasks with semantic stage labels and frame-level er- ror annotations.

**对 wiki 的映射**

- [paper-task-progress-tracking-errors-recovery-benchmark](../../wiki/entities/paper-task-progress-tracking-errors-recovery-benchmark.md)
