# 六项机器人与自动驾驶资料：核验进展

> 2026-11-06；本记录仅保存已查到的原始入口和去重线索，尚未完成六项 ingest，勿视为完整阅读或复现。

1. [An Unexpected Robot Policy](https://arxiv.org/abs/2609.24170)：摘要报告 Astra 在 RoboDojo 42 项任务、2,100 次试验中的平均成功率 22.48%；现有来源 `sources/papers/zhang_gpt6_astra_unexpected_robot_policy_2609_24170.md` 和实体 `wiki/entities/paper-gpt-6-astra-embodied-policy.md`。论文全文尚待读取核验。
2. [Astranomy](https://research.nvidia.com/labs/srl/projects/astranomy/)：NVIDIA SRL 技术报告，现有实体 `wiki/entities/astranomy.md`。需逐节核验原文、代码开放状态和运行入口。
3. [Waymo Safety Impact](https://waymo.com/safety/impact/)：官方安全影响页面。应核对报告日期、里程、碰撞定义、对照基准与统计限制后再整理。
4. [Alpamayo 2 Super](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super)：NVIDIA 技术博客（2026-08-04）。页面介绍 34B 驾驶 VLA；应完整核对模型、数据和代码以及开放与闭环评测口径。
5. [HomeBody](https://tml.stanford.edu/homebody/)：现有实体 `wiki/entities/paper-homebody.md`；需核对项目页、代码及更新状态，避免重复。
6. [Liber-0](https://www.liberai.com/blog/liber0/)：官方博客现可通过网页检索读取；已有待核验归档 `sources/sites/liber-0.md`。文章描述联合预测未来帧与动作的世界动作模型：60 个目标任务各 10 次，成功率 36.0%；其零样本指未针对目标任务收集机器人示范。机器人后训练后的内部进度分数与该成功率是不同指标，不能混同。模型、实验其余部分和开放状态仍需逐项核验。

## 待完成

通读各项主要原文及必要附录，复核现有节点和重复 PR；在各项目现有实体上补充基于原文的 Mermaid 流程图。官方存在可运行实现时阅读源码与入口，补充源码时序图；未开放时写明边界。完成后运行 `make ci-preflight`，修复检查后方可解除草稿状态。
