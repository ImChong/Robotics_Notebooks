# SequenceLink · 官方仓库核对

- **类型：** repo
- **原文：** https://github.com/General-Sequences/sequence-ai
- **项目页：** [General Sequences 官方站点](../sites/general-sequences-sequencelink.md)（https://generalsequences.com/）
- **技术报告：** [Scaling Robot Policies over Wide-Area Networks](../papers/gs-wan-scaling-robot-policies.md)
- **机构：** General Sequences
- **固定版本：** [0.66.3 / 05fec493005f7d76708afb00deca73c59e1ca87e](https://github.com/General-Sequences/sequence-ai/tree/05fec493005f7d76708afb00deca73c59e1ca87e)
- **许可证：** Apache-2.0（仓库代码）；策略、权重、第三方资源与托管服务单独处理。
- **核对日期：** 2026-11-28
- **核对范围：** 固定版本 README、Quickstart、推理/网络组件清单、动作余量调度、链路测量和开源/托管边界文档；核对策略与机器人 Quickstart、`seq/serve_local.py`、`sequence_ai/__init__.py` 和推理服务入口。官网仅用于核对托管产品边界，不代替技术文档和源码。
- **通读：** 已完成

| 官方入口 | 可核验范围 |
|---|---|
| [Quickstart](https://github.com/General-Sequences/sequence-ai/blob/05fec493005f7d76708afb00deca73c59e1ca87e/docs/quickstart.md)与[示例策略](https://github.com/General-Sequences/sequence-ai/blob/05fec493005f7d76708afb00deca73c59e1ca87e/examples/quickstart_policy/policy.py) | 占位策略从图像均值构造 10×7 数组；示例只保留 8 个动作步，不附真机驱动。 |
| [机器人示例](https://github.com/General-Sequences/sequence-ai/blob/05fec493005f7d76708afb00deca73c59e1ca87e/examples/quickstart_robot.py) | 首次等待服务就绪；后台请求下一动作块；根据经过时间跳步；`move_user` 与 `hold_user` 为调用方实现。 |
| [本地 CLI 服务](https://github.com/General-Sequences/sequence-ai/blob/05fec493005f7d76708afb00deca73c59e1ca87e/seq/serve_local.py)与[推理服务](https://github.com/General-Sequences/sequence-ai/blob/05fec493005f7d76708afb00deca73c59e1ca87e/sequence_ai/engine/serve.py) | 从策略目录实例化已注册的类和适配器，用 sample 观测驱动预热及优化核对；支持单请求，也支持已声明的批处理。 |
| [Slack 调度](https://github.com/General-Sequences/sequence-ai/blob/05fec493005f7d76708afb00deca73c59e1ca87e/docs/slack.md)与[链路测量](https://github.com/General-Sequences/sequence-ai/blob/05fec493005f7d76708afb00deca73c59e1ca87e/docs/how-do-i-measure-the-link-between-my-robot-and-my-gpu.md) | 用动作剩余时长、请求外部耗时和实际批耗时判断容量；不加载模型的 bench 测链路，不能代替策略端到端回放。 |
| [开放范围](https://github.com/General-Sequences/sequence-ai/blob/05fec493005f7d76708afb00deca73c59e1ca87e/docs/what-is-open-source-and-what-is-hosted.md) | Apache-2.0 数据面可自托管；区域 GPU 放置、机器生命周期、托管部署与计费属于服务控制面。 |
