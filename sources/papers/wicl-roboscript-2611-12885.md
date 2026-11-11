# World In-Context Learning for Agentic Robot Manipulation (WICL / RoboScript)

- **中文标题：** 面向智能体式机器人操作的世界上下文学习
- **论文：** <https://arxiv.org/abs/2611.12885>
- **PDF：** <https://arxiv.org/pdf/2611.12885>
- **项目：** <https://ArcheOne-AI.github.io/RoboScript/>
- **代码：** <https://github.com/ArcheOne-AI/RoboScript>
- **核验日期：** 2026-11-11
- **核对范围：** arXiv:2611.12885v1 33 页 PDF：摘要、正文 §1–7、算法/框图/实验/消融及附录 A–F 的数据设置、技能、原始结果、统计与评测协议，参考文献；对照 RoboScript README、quickstart_mock/run.py、RoboTwin run_pipeline.py 等官方入口。
- **通读：** 已完成

## 方法

RoboScript 用冻结基础模型作为 agent，而不训练专门动作策略：先抽取可重用技能（S1），再以每任务**一条演示**在模拟世界反复试演（S2），把结果写入 **Task Memory** 而非模型权重；部署与官方测试时仅暴露实际机器人可获得的观测与工具（P3），并可后续反思更新。模拟评判所需的 ground truth 由控制面持有，不应泄漏到行动代理可用提示。

## 结果与边界

- 官方 RoboTwin 2.0 共 **50 项双臂任务**，作者报告平均成功率 **83.4%**；clean **83.6%**，randomized **83.2%**。对照最强训练策略用 50 演示/任务得到 **61.7%**。
- 项目展示模拟 ALOHA 记忆映射到真机双臂 PiPER 的示范；仓库 README 同时将 **Real-robot deployment code** 列为 TODO。真机视频不意味着相应执行/安全代码完整开源。
- 源码可运行 mock smoke test 和 RoboTwin 管线，但仍需网关凭据、RoboTwin 资产、模型与运动服务。

## 全文核验补充（2026-11-11）

### 世界上下文学习机制

- **S1** 从机器人演示构建可复用 primitives；**S2** 在干净场景模拟试演并通过控制面评判整理 **Task Memory**；**P3** 新场景只开放真实机器人可用的动作与观测工具。模型权重冻结，反思可以继续改善记忆，不是 PPO / SFT 训练。
- 控制面可以用真值打分，但不能让部署行动者从 prompt 或服务接口直接读到 ground truth。任务记忆是 agent 可读的结构化经验，不等于世界模型参数。
- 官方框架在 mock 环境有可运行 smoke test，RoboTwin 2.0 有集成执行；真机视频展示双臂 PiPER，但 README 的部署代码仍列 TODO。

### 实验协议与局限

- RoboTwin 2.0 **50 项双臂操作**，官方报告 Clean **83.6%**、Randomized **83.2%**、合计 **83.4%**；每任务每设定 10 episodes。与最优训练策略 **61.7%** 的对比基于不同演示、计算和评测预算，不可视为严格等资源 ablation。
- 真实 PiPER 四个任务分别 **10/10、7/10、7/10、6/10** 的成功样例；该样本规模不能推断未见任务大规模零样本能力。
- 消耗主要转移到长时间多轮试演、模型工具调用和仿真/运动规划运行时；单条示范不等于一次模型调用就能完成全部训练。

## 对 wiki 的映射

- [WICL / RoboScript 主节点](../../wiki/entities/paper-world-in-context-learning-roboscript.md)
