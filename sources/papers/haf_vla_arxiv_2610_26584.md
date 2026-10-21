# HAF-VLA：人形视觉-语言-动作控制的层次化动作流框架（arXiv:2610.26584）

> 来源归档（ingest）

- **标题：** HAF-VLA: Hierarchical Action Flow Framework for Humanoid Vision-Language-Action Control
- **类型：** paper / vla / humanoid / loco-manipulation / flow-matching
- **arXiv abs：** <https://arxiv.org/abs/2610.26584>
- **arXiv HTML：** <https://arxiv.org/html/2610.26584>
- **PDF：** <https://arxiv.org/pdf/2610.26584>
- **项目页：** [HAF 官方项目页](https://grange007.github.io/HAF/)（与已收录的 HAF 框架同一项目）
- **代码：** 截至 2026-10-21，官方项目页未列出代码仓库或权重链接；沿用既有 HAF 开源状态
- **与既有条目的关系：** 同一团队 HAF（arXiv:2608.16837）的 HAF-VLA 子框架论文，不新建重复项目节点
- **一句话说明：** 在预训练 flow-matching VLA 上，以跨阶段 KV cache 条件化的三阶段去噪逐步生成全身动作，先 locomotion 与 head，再 waist，最后全身 manipulation。

## 摘要级要点

- **问题：** 人形 loco-manipulation 需联合协调移动、姿态、视线和双臂，底座或腰部动作不准会导致上身补偿过度并破坏稳定性。
- **架构：** 继承预训练 flow-matching VLA 的视觉-语言前缀编码器和 action expert 权重，通过可学习 stage embedding 与跨阶段 KV cache 适配全身动作。
- **阶段顺序：** 共享全身 action space 逐步去噪：locomotion + head → waist refinement → full-body manipulation；每阶段读取此前阶段 action 表征并细化输出。
- **证据：** 论文报告六项真实世界任务实验，并称该框架提高已有模型在人形 loco-manipulation 上的表现；这里不沿用旧版 HAF 七任务成绩作为本论文成绩。
- **与既有 HAF 的关系：** 该论文聚焦 HAF-VLA 的动作生成框架；原 HAF 论文还包含 HAF-Steer 频谱潜空间离线–在线 RL 后训练。

## 对 wiki 的映射

- [HAF 实体页](../../wiki/entities/paper-haf-humanoid-vla-adaptation.md) — 与 arXiv:2608.16837 合并维护
- [VLA](../../wiki/methods/vla.md) — flow-matching 策略的人形全身动作生成
- [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 移动与操作耦合任务
