# TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks

- **论文：** <https://arxiv.org/abs/2611.22115v1>
- **全文：** [HTML](https://arxiv.org/html/2611.22115v1) · [PDF](https://arxiv.org/pdf/2611.22115v1)
- **作者：** Xingchen AGI Lab, China Telecom（个人作者见附录 A）
- **机构：** 中国电信人工智能（TeleAI）
- **版本：** v1，2026-11-17
- **核对范围：** 全文 §1–6，架构与推理表 1–6、预训练剖析与消融、TeleGym 与部分轨迹续跑、CPU 隔离评测及附录 A；与 Preview 模型卡和代码仓库互证发布边界。
- **通读：** 已完成
- **模型：** [Xing4.1-Preview 发布卡与权重](../models/xing4-1-preview.md)
- **代码：** [部署文档及 vLLM 插件](../repos/xing4-1-preview.md)

## 核读要点

1. 报告把 29B/4B 的 Xing4.0 公开成员与 285B/17B 的主要研究配置分别列出，不能把其原生上下文长度、注意力结构或权重混用。Preview 后续卡片确认大模型的发布身份与配置。
2. 285B 注意力配置的 4.68 倍为预填充仿真对照，DSpark 的 2.67–3.77 倍为表 5 的输出吞吐比。前者有 256K 上下文和 32 卡条件；不代表默认开源服务端到端速度。
3. 预训练同时优化超节点分片与通信重叠、Muon 单 owner、激活重算、文档感知 KV 窗口和主机延迟。表 7–11 有明确的不同基线，不应跨行累加收益。
4. TeleGym 管理任务环境和会话，训练网关构建 token 级轨迹；部分 rollout 可跨策略更新续跑，保存行为策略概率，对尚未完成的轨迹不提前赋终止奖励。

**对应 wiki：** [Xing 4 技术报告](../../wiki/entities/paper-telechat-xing4.md)
