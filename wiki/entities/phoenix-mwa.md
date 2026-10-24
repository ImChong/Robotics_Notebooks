---
type: entity
tags: [world-action-models, robot-manipulation, tactile-sensing, reinforcement-learning, anyverse-dynamics]
status: complete
updated: 2026-10-24
project_id: anyverse-phoenix-mwa
project: https://openphoenix-team.github.io/Phoenix-pages/
code: https://github.com/Anyverse-Dynamics/Phoenix-MWA
related:
  - ../concepts/world-action-models.md
  - ../methods/model-based-rl.md
  - ../methods/imitation-learning.md
  - ../entities/paper-vt-wam-visuotactile-contact-rich.md
sources:
  - ../../sources/sites/phoenix-mwa.md
  - ../../sources/repos/anyverse-dynamics-phoenix-mwa.md
summary: "Phoenix 是面向机器人操作的世界动作模型系统：联合预测未来视觉与动作，并用运动—触觉交互表征和学习到的视觉触觉模拟器进行后训练。"
---

# Phoenix-MWA：从世界预测到接触反馈操作

**Phoenix** 是 Anyverse Dynamics 与中科院自动化所强化学习团队联合开发的机器人操作学习系统，由视觉—动作预训练、触觉接触技能学习和世界模型中的强化学习后训练组成。

## 一句话定义

让机器人不仅从视频和示范中拟合动作，还预测动作后的视觉结果、学习接触变化，并在反馈中迭代改善策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MWA | 官方未展开 | Phoenix-MWA 的世界动作预训练主干；项目页未说明缩写全称 |
| ILA | Interaction Latent Action | 将运动与触觉变化编码成策略生成目标的交互潜变量 |
| GRPO | Group Relative Policy Optimization | 按同一初始条件下的多条 rollout 相对表现更新策略 |
| UMI | Universal Manipulation Interface | 项目预训练数据中的一种人类操作记录来源 |
| RTC | Real-Time Chunking | 执行动作块时并行准备后续动作以维持连续性 |

## 为什么重要

纯视觉轨迹看起来合理，并不保证工具持续贴合物体，也不保证插入或擦拭等动作真正成功。Phoenix 把问题拆成三层：跨任务预训练提供初始策略；触觉交互表征补上接触信息；学习到的环境模型提供动作结果反馈，用于后训练。

## 核心方法栈

1. **Phoenix-MWA 预训练：** 输入多视角 RGB、机器人状态、语言指令和本体提示，联合预测未来视觉状态与动作。项目页称数据池包含约 54,413 小时原始视频、约 39,306 小时处理后数据和约 33.8 亿帧。
2. **Phoenix-ILA 接触后训练：** 从同步的运动与触觉示范学习潜变量，并以它替代直接动作目标；推理时使用当前观测与历史触觉，不要求未来触觉。
3. **Phoenix-PhysCanvas 交互后训练：** 策略提出动作，视觉—触觉世界模型预测结果，奖励模型评分，再用 GRPO 更新策略；更新结果返回物理机器人评估。
4. **闭环执行：** 机器人执行短动作前缀，重新观测后再规划，而不是一次输出完整长任务动作。

## 实验与评测

官方技术报告在各基准对应的后训练配置下报告以下成功率：

| 基准 | 报告成功率 |
|------|-----------:|
| LIBERO-Plus | 94.63% |
| LIBERO-Pro | 66.48% |
| RoboCasa-GR1 | 82.67% |
| RoboTwin 2.0 C2R | 72.42% |
| Bench2Dex | 45.60% |
| RoboDojo-RealWorld | 26.11% |

接触技能实验中，加入 ILA 后，削皮成功率从 20% 提升到 40%，擦拭从 75% 提升到 90%。PhysCanvas 的 GRPO 后训练在试管插入任务上，三种实测条件的平均成功率从 58.33% 升至 75%；积木操作从 80% 升至 86.67%。这些是真机结果，但仅覆盖报告中的任务和条件。

## 工程实践与开源状态

- 代码仓库 main 分支公开 MWA 预训练实现；项目页也链接模型权重入口。
- README 指出后训练和基准评测在 post-train 分支；编码器、数据处理和 RTC 部署代码尚标为待开放。
- 完整训练还依赖单独准备的权重、manifest、文本缓存与外部数据路径。克隆仓库本身不足以复现训练。
- 复现实验时应按项目报告的基准配置分别记录后训练阶段和评估设置，避免把不同协议下的成功率当成统一零样本结果。

## 局限与风险

- 世界模型给出的视觉—触觉结果是学习到的预测，不是真实物理反馈；预测偏差可能误导强化学习更新。
- 项目页明确指出 PhysCanvas 的真机强化学习收益目前展示在试管插入和积木两项任务，更多任务上的迁移与稳定性仍待评估。
- 公开代码、模型入口和数据的开放程度不同；目前更准确的描述是**部分开源**，不能据此推断完整训练和部署栈可直接复现。
- 技术报告的榜单数字依赖各基准自己的后训练和评测配置，横向对比需要先核对协议。

## 关联页面

- [World Action Models（世界–动作模型）](../concepts/world-action-models.md)
- [Model-Based RL（模型式强化学习）](../methods/model-based-rl.md)
- [Imitation Learning（模仿学习）](../methods/imitation-learning.md)
- [VT-WAM：视觉触觉接触丰富操作](./paper-vt-wam-visuotactile-contact-rich.md)

## 参考来源

- [Phoenix 官方项目页与技术报告](../../sources/sites/phoenix-mwa.md)
- [Phoenix-MWA 官方代码仓库归档](../../sources/repos/anyverse-dynamics-phoenix-mwa.md)

## 推荐继续阅读

- [官方技术报告与真机演示](https://openphoenix-team.github.io/Phoenix-pages/index_zh.html)
- [官方代码仓库](https://github.com/Anyverse-Dynamics/Phoenix-MWA)
- [官方模型入口](https://huggingface.co/AnyverseDynamics)
