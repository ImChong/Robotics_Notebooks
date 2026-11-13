# [2026-11-13] ingest | 星尘智能 CoMind / Be My Eye

- **操作：** ingest。
- **意图：** 收录星尘智能 CoMind 项目页及 arXiv:2611.12295v1 论文，准确组织双机器人受限通信下的任务相关上下文共享，并对比零通信 CHORUS。
- **已核对：** arXiv HTML v1 §1–6 方法、公式、Tables 1–5、定性实验与限制；官网 CoMind 任务描述与公开资源入口。
- **新增主实体：** [CoMind / Be My Eye](../wiki/entities/paper-be-my-eye-comind.md)；包含双语标题、机制、论文方法 Mermaid 流程图、数值证据和开源边界。
- **来源：** [论文](../sources/papers/be-my-eye-arxiv-2611-12295.md)、[官网](../sources/sites/astribot-comind-project.md)、[官方代码核验](../sources/repos/astribot-comind-code-status.md)。
- **开源判定：** 截至本日官网未公开该方法可运行代码、权重与数据；不虚构源码时序图。
- **关联：** 新主实体引用 [CHORUS](../wiki/entities/paper-chorus.md) 作零通信对照；[人形多机协调](../wiki/concepts/humanoid-multi-robot-coordination.md) 增加 CoMind 入链。
- **通读补齐：** PR #3699 复核中逐篇通读 Swarm Sensors 2025、SPL 2401.15026v1、ARTEMIS 2512.09431v1、RoboCompiler-Humanoids 2610.14237v1 及 CoMind 2611.12295v1；前四篇集中为[一手论文全文核验记录](../sources/papers/humanoid-coordination-fulltext-review-2026-11-13.md)，协调概念页以 `original` 关联该记录与 CoMind 归档。
- **纠错：** 区分单机内管理与队级集中控制、SPL 定性赛事与 60 min 仿真、Swarm 控球百分点与缺员/丢包口径；CoMind 图中未来帧改为独立训练目标，补充 Process Score 权重定义。
