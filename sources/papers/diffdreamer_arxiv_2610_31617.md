# DiffDreamer：世界模型与可微仿真耦合的四旋翼控制

- **论文：** Fanxing Li, Yuxiang Huang, Shengyang Wang, Fangyu Sun, Shuyu Wu, Tianbao Zhang, Dexin Zuo, Yan Yufei, Danping Zou, Wenxian Yu, *DiffDreamer: Coupling World Models with Differentiable Simulation for Agile Quadrotor Control*。
- **原文：** [arXiv:2610.31617v1](https://arxiv.org/pdf/2610.31617v1)，2026-10-22；已核对全文方法、实验和局限。
- **来源状态：** PDF 页眉为 IEEE Robotics and Automation Letters；正式发表信息以出版方记录为准。
- **源码：** 截至 2026-11-03 未找到可确认的官方 DiffDreamer 实现。

## 资料摘录与映射

1. RSSM 隐状态预测环境变化，可微模拟器传播自身物理状态，两路使用相同动作同步展开；映射至 [DiffDreamer 实体](../../wiki/entities/paper-diffdreamer-quadrotor.md) 的流程总览。
2. 不可微任务奖励由 critic 估值，可微运动奖励通过物理模型反传；映射至实体的核心机制。
3. 在跟踪及动态门竞速仿真中与 RL 和传统控制基线比较；真机在动捕反馈及经辨识的动力学条件下验证；映射至实体的评测与局限。
