# Scaling Robot Policies over Wide-Area Networks · GS-WAN 官方技术报告

- **类型：** paper
- **原文：** https://generalsequences.com/papers/scaling-robot-policies-over-wide-area-networks.pdf
- **代码：** https://github.com/General-Sequences/sequence-ai
- **项目页：** https://generalsequences.com/
- **署名：** General Sequences（报告建议引用名），作者 Yongzhe Wang。
- **版本：** 2026 年 11 月公开技术报告，PDF 44 页；仓库代码快照另见 [SequenceLink](../repos/sequence-ai.md)。
- **核对日期：** 2026-11-28
- **核对范围：** PDF 的动作块期限定义、GS Inference、GS Network、路由、评测、局限、结论与配套附录。延迟/容量数据按报告表格的模型、GPU、网络、动作期限与场景解释，不外推为真机结果。
- **通读：** 已完成

- §2–3：机器人发出下一请求时，尚未执行的有效动作定义时间预算。批处理提高吞吐，但等待与传输也消耗同一预算，因此不能仅用 QPS 做容量决策。
- §3.5 表 4：报告给出单 A100-40GB 的 GR00T N1.6 CUDA Graph 条件，单条执行至少保持 9 个仿真机器人按期，批处理为 24。这里的 9 是下界；报告未测这些机器人在实体工作站的任务成功率。
- §4–5：端侧图像缩放、连接保温等措施针对传输附加耗时；区域路由与备用连接另属托管服务设计。该结果不能当成任意线路或任意无线链路的保障。
- §6–7：LIBERO-10 的延迟注入是仿真试验；报告还测了美国网络路径上的服务时延，覆盖 GPU 为 L4 和 A100。调度按时率与操作成功率分别报告。

对 wiki 的映射：[SequenceLink](../../wiki/entities/sequencelink.md)，作为本次收录仓库的配套报告，说明公开数据面与整套 GS-WAN 的边界。
