# LEASH: Stay Safe in Agentic Robotics｜分层安全门保障机器人智能体

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14115
- **PDF：** https://arxiv.org/pdf/2611.14115
- **核对范围：** 核对论文 arXiv v1 摘要、用户 2026-11-12 资讯附件、官方源码 README 与公开入口；未通读全文方法、实验与附录，未执行实验。
- **通读：** 未完成
- **收录日期：** 2026-11-14
- **代码或数据：** https://github.com/zhuqinfeng1999/LEASH

## 核心摘录

1. RoboRisk 覆盖任务拒绝、计划修复、执行危险、注入、顺序记忆五类风险，92 项仿真与 13 项真机任务。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-leash.md)
2. 规划器和三个安全门共享统一安全规则，以实时摄像头上下文判断执行风险。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-leash.md)
3. 同时看 Harm Rate 和 Safe Success Rate，安全拒绝危险任务也可视为成功。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-leash.md)

## 作者报告的结果与限制

文章报道 GPT-6 Astra 仿真中伤害率 11.23%→0%、安全成功率 78.99%→80.80%（每配置 276 次）；真机每配置 13 次，未观察到伤害不能证明普遍安全。

依赖审查模型与视觉感知；未知危害可能漏检，未独立复现实验。

## 待补

需通读论文图表、正文和附录并逐项校验结果；当前内容为原始摘要及资讯整理，不替代原文。
