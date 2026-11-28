# SafeManip：机器人操作时序安全评测 / A Property-Driven Benchmark for Temporal Safety Evaluation in Robotic Manipulation

- **作者：** Chengyue Huang、Khang Vo Huynh、Mellon M. Zhang、Sebastian Elbaum、Zsolt Kira、Lu Feng。
- **机构：** Georgia Institute of Technology、University of Virginia。
- **论文：** [arXiv:2605.12386v3](https://arxiv.org/abs/2605.12386v3) · [HTML](https://arxiv.org/html/2605.12386v3) · [PDF](https://arxiv.org/pdf/2605.12386v3)
- **核对范围：** v3 全文及附录，重点 §3–5、图 5–7 与附录 D 的机会计数和失效分析；比对官方 README、离线监控器入口与实际函数。
- **通读：** 已完成
- **代码：** [chengyuehuang511/SafeManip](https://github.com/chengyuehuang511/SafeManip)（固定快照 `cb9ca772c5a2e7fce9a0d44f966aabd3e2f5eb3b`；[实现归档](../repos/safemanip.md)）
- **入库日期：** 2026-11-28

1. §3 将仿真状态映射到带语义绑定的布尔谓词，以有限轨迹时序逻辑表达先决条件、释放稳定、交叉污染等八类过程义务，并跟踪重复触发。
2. §4 以触发数为公共分母：事件违例为 V/N，违例时间步暴露为 U/N；汇总先合并计数，不将任务成功混入评分。无触发表示规则没有被检验。
3. §5 在 RoboCasa365 50 项及 LIBERO 40 项仿真任务使用七个模型家族计十种策略／训练变体（各设定并非跨两仿真套件完全对称）；每任务每受测设定 50 回合。失败回合每触发风险较成功回合高 1.7–7 倍，但成功回合仍有规则违例。

- **对 wiki 的映射：** [SafeManip](../../wiki/entities/paper-safemanip.md)
