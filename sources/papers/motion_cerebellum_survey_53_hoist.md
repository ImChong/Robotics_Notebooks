# HOIST 人形机器人悬挂负载操作的模仿学习与高效微调

> 来源归档（ingest · 运动小脑 64 篇长文 第 53/64）

- **标题：** HOIST 人形机器人悬挂负载操作的模仿学习与高效微调
- **类型：** paper
- **运动小脑分类：** H 真实任务
- **机构：** 佛罗里达大学
- **项目页：** https://arxiv.org/abs/2606.00252v1
- **入库日期：** 2026-06-18
- **一句话说明：** 任务：悬挂负载操作考验后果建模。输入是 VR 示教、悬挂负载状态和机器人本体状态；实现上先训练高层任务策略，再用 batched RL 对自主 rollout 做样本高效微调；核心是处理负载摆动、滞后和反作用力对全身平衡的影响。

## 核心摘录（策展，非全文）

- **在动作小脑地图中的位置：** H 真实任务，编号 **53/64**。
- **公众号站位：** 任务：悬挂负载操作考验后果建模

## 对 wiki 的映射

- [paper-motion-cerebellum-hoist](../../wiki/entities/paper-motion-cerebellum-hoist.md)
- [motion-cerebellum-category-08-real-tasks](../../wiki/overview/motion-cerebellum-category-08-real-tasks.md)

## 参考来源（原始）

- 项目页：https://arxiv.org/abs/2606.00252v1
- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)

## 项目页与开源状态核查（2026-07-22）

- **论文：** arXiv:2606.00252。
- **代码：** 未确认 HOIST 官方可运行代码；论文引用 GR00T Whole-Body Control 作为固定低层执行栈。
- **关键数字：** HOIST 用 50 demos + 30 RL rollouts；真实平台 `|Δx|+|Δy|` 从 VLA-50 的 9.28 cm 降到 6.38 cm；相对 pure VLA translational error 降 19.9 cm、raw angular error 降 3.56°。
- **wiki 深化：** [paper-motion-cerebellum-hoist](../../wiki/entities/paper-motion-cerebellum-hoist.md)。
