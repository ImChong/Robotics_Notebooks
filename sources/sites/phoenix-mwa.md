# Phoenix 项目页（机器人操作世界动作模型）

- **标题：** Phoenix: Scaling Robotic Manipulation through Visual and Interaction Representations
- **类型：** site / project-page / technical-report
- **URL：** <https://openphoenix-team.github.io/Phoenix-pages/index_zh.html>
- **代码：** <https://github.com/Anyverse-Dynamics/Phoenix-MWA> — 预训练实现
- **模型：** <https://huggingface.co/AnyverseDynamics>
- **项目组：** Anyverse Dynamics（无界动力）与中科院自动化所强化学习团队
- **报告日期：** 2026-10-13（项目页标注）
- **入库日期：** 2026-10-24
- **配套仓库：** [Anyverse-Dynamics/Phoenix-MWA](../repos/anyverse-dynamics-phoenix-mwa.md)
- **沉淀到 wiki：** [Phoenix-MWA](../../wiki/entities/phoenix-mwa.md)

## 一句话摘要

Phoenix 将多视角视觉、机器人状态与语言条件下的未来视觉状态和动作联合预测，与示范中的运动—触觉接触表征、世界模型中的强化学习后训练结合，面向真实机器人操作。

## 项目页要点

- **Phoenix-MWA：** 以大规模机器人示范、第一人称视频和 UMI 记录预训练世界动作模型；执行动作块前缀后重新观察并规划。
- **Phoenix-ILA：** 从同步运动与触觉示范学习交互潜变量，使策略目标同时包含运动和触觉变化。
- **Phoenix-PhysCanvas：** 从真实机器人轨迹学习视觉—触觉反馈模型，在模型预测的交互结果上采样 rollout，再以任务奖励和 GRPO 更新策略。
- **评测口径：** 项目页报告六项操作基准，并说明各基准采用各自后训练和评估配置；成功率不可脱离对应设置直接横向比较。
- **真机证据：** 项目页展示多步操作、削皮、擦拭、试管插入和积木操作；PhysCanvas 强化学习后的收益在两类任务上回到真机评估。

## 源码与资源开放核查（2026-10-24）

| 资源 | 状态 | 核查结论 |
|------|------|----------|
| 预训练代码 | 已发布 | 官方 GitHub 仓库提供 Phoenix-MWA 预训练实现 |
| 后训练与基准代码 | 部分发布 | README 指向 post-train 分支；编码器预训练、数据处理和 RTC 部署分支在 README 中仍标注 coming soon |
| 模型权重 | 有官方入口 | Hugging Face 组织页由项目页链接；实际可用模型以页面当前内容为准 |
| 数据 | 部分可复现 | 仓库 README 指出原始视频与数值轨迹位于外部共享存储，Git 克隆不包含完整 resources |

因此按**部分开源**记录：公开仓库不等于一键复现实验；需另行准备资源、权重及数据访问。

## 关联资料

- 官方代码：[Anyverse-Dynamics/Phoenix-MWA](../repos/anyverse-dynamics-phoenix-mwa.md)
- 知识节点：[Phoenix-MWA](../../wiki/entities/phoenix-mwa.md)
