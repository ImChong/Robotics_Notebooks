# World In-Context Learning for Agentic Robot Manipulation (WICL / RoboScript)

- **中文标题：** 面向智能体式机器人操作的世界上下文学习
- **论文：** <https://arxiv.org/abs/2611.12885>
- **PDF：** <https://arxiv.org/pdf/2611.12885>
- **项目：** <https://ArcheOne-AI.github.io/RoboScript/>
- **代码：** <https://github.com/ArcheOne-AI/RoboScript>
- **核验日期：** 2026-11-11
- **核对范围：** 论文摘要、方法四阶段、RoboTwin 50 任务结果、官方 README、quickstart_mock/run.py 与 RoboTwin run_pipeline.py；论文其余附录尚待逐页复核。
- **通读：** 未完成

## 方法

RoboScript 用冻结基础模型作为 agent，而不训练专门动作策略：先抽取可重用技能（S1），再以每任务**一条演示**在模拟世界反复试演（S2），把结果写入 **Task Memory** 而非模型权重；部署与官方测试时仅暴露实际机器人可获得的观测与工具（P3），并可后续反思更新。模拟评判所需的 ground truth 由控制面持有，不应泄漏到行动代理可用提示。

## 结果与边界

- 官方 RoboTwin 2.0 共 **50 项双臂任务**，作者报告平均成功率 **83.4%**；clean **83.6%**，randomized **83.2%**。对照最强训练策略用 50 演示/任务得到 **61.7%**。
- 项目展示模拟 ALOHA 记忆映射到真机双臂 PiPER 的示范；仓库 README 同时将 **Real-robot deployment code** 列为 TODO。真机视频不意味着相应执行/安全代码完整开源。
- 源码可运行 mock smoke test 和 RoboTwin 管线，但仍需网关凭据、RoboTwin 资产、模型与运动服务。

## 对 wiki 的映射

- [WICL / RoboScript 主节点](../../wiki/entities/paper-world-in-context-learning-roboscript.md)
