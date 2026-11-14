# LEASH: Stay Safe in Agentic Robotics｜分层安全门保障机器人智能体

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14115
- **PDF：** https://arxiv.org/pdf/2611.14115
- **核对范围：** arXiv HTML v1 正文 §1–6 和附录 A–D 的基准构成、三道安全门、各模型结果与成本；核对官方 README；未独立运行模拟器。
- **通读：** 已完成
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

## 原文核对摘录（2026-11-14）

论文 §3–4：RoboRisk-S 有 92 项任务，五类为 Task-level Refusal、Plan Repair、Execution-time Hazard、Injection、Sequential Memory；每规划器配置 3 种种子共 276 次。Task Gate 可以终止危险任务，Action Gate 拒绝动作后重规划，Commit Gate 在长技能执行中检查预先标记的关键操作，当前仅监视夹爪张开与倒液。§5 中 GPT-6 Astra 仿真 HR 11.23%→0%，SSR 78.99%→80.80%；物理 RoboRisk-R 每配置仅 13 次。附录提供任务清单、门策略、跨模型结果与风险分析。作者报告少量时间开销，但不代表语义决策防护能提供形式化安全保证。

## 复现边界

仍未独立运行训练、评测与机器人实验，论文数字为作者报告。对 wiki 的映射：[论文详情](../../wiki/entities/paper-leash.md)。
