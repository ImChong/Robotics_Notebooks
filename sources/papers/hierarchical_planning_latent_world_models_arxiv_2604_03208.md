# Hierarchical Planning with Latent World Models（arXiv:2604.03208）

> 来源归档（paper / HWM）

- **标题：** Hierarchical Planning with Latent World Models
- **arXiv：** <https://arxiv.org/abs/2604.03208>
- **HTML：** <https://arxiv.org/html/2604.03208v2>
- **项目页：** <https://kevinghst.github.io/HWM/>
- **代码：** <https://github.com/kevinghst/HWM_PLDM>
- **作者：** Wancong Zhang、Basile Terver、Artem Zholus、Soham Chitnis、Harsh Sutaria、Mido Assran、Randall Balestriero、Amir Bar、Adrien Bardes、Yann LeCun、Nicolas Ballas
- **机构：** Meta FAIR；纽约大学；Mila；布朗大学（以论文 HTML 署名为准）
- **版本：** v2，2026-06-16
- **许可：** arXiv 页面列 CC BY 4.0
- **入库日期：** 2026-10-06
- **一句话说明：** 在共享 latent 空间训练多个时间尺度的 world model，高层以 latent macro-action 规划子目标，低层以原始动作追踪子目标。

## 论文要点

HWM 是在视觉 latent world model 上进行分层 model predictive control（MPC）的规划范式。高层模型在较长时间尺度上预测 latent waypoint；低层模型根据当前状态和原始动作序列接近第一个 waypoint；执行后再观测并滚动重规划。论文使用学习到的 action encoder 将一段低层动作压成 macro-action。

论文在 Franka 双臂/夹爪实验中以单张目标图像指定目标，报告 pick-and-place 任务 70% 成功率，对照同设定的单层 V-JEPA 2-AC planner 为 0%。另外在 Push-T 和迷宫任务报告长时程规划提升，特定对比中规划计算最多减少约 3 倍。所有数字都限定在论文任务、数据和试验设置内。

## 工程阅读重点

- HWM 将潜空间预测和在线 MPC 结合；不是单一端到端策略网络。
- 高层动作是对低层动作片段的潜变量压缩，子目标由高层 rollout 产生。
- 低层和高层规划都需要预测误差可控；高层 latent 子目标过粗会丢失低层精细控制所需信息。
- 真机证据来自 Franka，不应直接推断到人形机器人或复杂全身控制。

## 对 wiki 的映射

- [HWM 论文实体页](../../wiki/entities/paper-hwm-latent-world-model-planning.md)
- [VideoDB JEPA 长文实体页](../../wiki/entities/article-videodb-jepa-world-models.md)
- 站内已有：[LeWorldModel](../../wiki/entities/paper-lewm.md)、[V-JEPA 2.1](../../wiki/entities/paper-sa-2603-14482-v-jepa-2-1-unlocking-dense-features-in-video-sel.md)
