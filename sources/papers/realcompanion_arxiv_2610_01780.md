# RealCompanion: Benchmarking Human Understanding from Reasoning over Longitudinal Real-World Conversations

> 来源归档（paper）

- 类型：paper / benchmark / conversational memory
- arXiv：https://arxiv.org/abs/2610.01780v3
- 作者：Arman Behnam、Sunglyoung Kim、Liangwei Yang
- 入库日期：2026-10-11
- 说明：使用真实纵向 AI companion 对话，评测用户理解、记忆触发和消息级证据追溯；arXiv 页面未列独立代码仓库链接。

## 摘录与 wiki 映射

- 数据：10 段真实人-AI companion 关系、27,218 条消息、最长 120 天；包含对话及 profile、persona、chat ground truth、question set 等派生信息，标注可回溯至原始消息。
- 发现：最近窗口在 95.9% probes 中能找到目标消息，但对真正需要记忆的 probe 覆盖仅 2.2%；所测 detector 不能可靠判定何时需记忆。显式标记 memory 令消息使用率提升 10–14 个百分点。
- 成本：三个 agent 系统的 persona F1 相同，但成本相差 31 倍。
- 对 wiki 的映射：[RealCompanion 实体页](../../wiki/entities/paper-realcompanion.md)，并与 [Self-Retrospection Distillation](../../wiki/entities/paper-self-retrospection-distillation.md) 互链。
