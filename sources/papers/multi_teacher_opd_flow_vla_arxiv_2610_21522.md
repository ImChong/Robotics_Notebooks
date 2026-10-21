# Multi-Teacher On-Policy Distillation for Flow-Based Vision-Language-Action Models

> 来源归档（paper）

- **标题：** Multi-Teacher On-Policy Distillation for Flow-Based Vision-Language-Action Models
- **中文标题：** 面向基于 Flow 的视觉语言动作模型的多教师 On-Policy 蒸馏
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.21522>
- **入库日期：** 2026-10-21
- **线索来源：** [公众号综述（33 篇 Manipulation）](../raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)（具身运控lab，2026-10-20）
- **一句话说明：** 两层 on-policy 蒸馏，把多个冻结的领域专家 Flow-VLA 合并进一个学生策略。
- **沉淀到 wiki：** [`wiki/entities/paper-multi-teacher-on-policy-distillation-flow-vla.md`](../../wiki/entities/paper-multi-teacher-on-policy-distillation-flow-vla.md)

## 摘要要点（arXiv 摘要页）

- 专门化 VLA 难以合并为一个共享策略；Flow-based VLA 的分布错配出现在观测与动作流中间状态两层。
- 方法：学生交互产生经验，并在其自身动作流轨迹上接受教师速度监督；从联合 SFT 初始化开始。

## 开源状态

- **未核查代码**：arXiv 摘要页未列代码链接，入库日未找到项目页。
