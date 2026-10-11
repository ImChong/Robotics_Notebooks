# Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight

> 来源归档（paper）

- 类型：paper / agent learning / distillation
- arXiv：https://arxiv.org/abs/2610.08077v2
- 作者：Haoxiang Zhang、Qinglin Chen、Hiroaki Hayashi、Zhuofeng Li、Siming Zhang、Jiaxin Zhang、Jixuan Chen、Fang Wu、Pan Lu、Silvio Savarese、Julian McAuley、Chien-Sheng Wu
- 机构：Salesforce AI Research 等
- 入库日期：2026-10-11
- 注意：SRD 在此指 Self-Retrospection Distillation；与库中 State–Readout Decoupling（同缩写）不是同一论文。

## 摘录与 wiki 映射

- RLVR 的 group-relative 信号在一组 rollout 全部同奖时消失，但轨迹仍体现任务需求和失败原因。
- 交互前策略预测所需知识/陷阱；完成 rollout 后 stop-gradient 副本利用轨迹形成 hindsight 监督，再蒸馏为原策略的事前 foresight。推理时无需额外 foresight 生成。
- 在 10 项工具推理和长时程 agent 任务上，论文报告最高提升 24.2 个百分点；37–98% rollout groups 奖励一致时仍获益。2B 设置中 98% group 全失败，RLVR 为 0% 成功，加入 SRD 为 60.6%。
- 对 wiki 的映射：[SRD 实体页](../../wiki/entities/paper-self-retrospection-distillation.md)，与 [RealCompanion](../../wiki/entities/paper-realcompanion.md) 和 [Cross-Tokenizer OPD](../../wiki/entities/paper-cross-tokenizer-opd.md) 互链。
