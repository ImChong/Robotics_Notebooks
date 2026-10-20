# Systematically Exploring the Capabilities of GPT-6 Astra as Embodied Policies

> 来源归档（arXiv 预印本）

- **标题：** Systematically Exploring the Capabilities of GPT-6 Astra as Embodied Policies
- **类型：** paper / evaluation
- **原始链接：** <https://arxiv.org/abs/2609.38537>
- **作者：** Galbot Team（Xuchuan Chen 等 33 位作者）
- **提交日期：** 2026-09-29（v1）
- **入库日期：** 2026-10-20
- **一句话说明：** 跨操纵、灵巧操作、移动操作、导航、运动与人形全身任务评估 Astra 作为具身策略，并报告控制能力与推理时延的明显落差。

## 核心摘录（面向 wiki 编译）

### 1) 跨任务评测设计

- **摘录要点：** 论文覆盖六个领域，比较直接控制、与已训练策略协作以及基于反馈的适应等使用方式；涉及 RoboDojo、DexJoCo、RoboCasa365、RxR、HM3D 与 HumanoidBench。
- **对 wiki 的映射：** [GPT 6 Astra 具身策略评测](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md) — 补充为跨领域评估来源，需与其十任务 RoboDojo 结果分开记录。

### 2) 混合控制结果

- **摘录要点：** 在所评估子集中，与 π0.5 混合的操纵成功率为 48%；DexJoCo 十次经验引导试验中混合控制成功率为 50%；RoboCasa365 移动操作混合控制成功率为 38.7%。这些数值仅适用于论文所选任务和设置。
- **对 wiki 的映射：** [π0.5](../../wiki/entities/paper-pi05-open-world-vla.md) — 记录通用模型与已训练策略协作的案例，保留基准与样本量。

### 3) 可部署性与限制

- **摘录要点：** Astra 在 RxR 指令跟随与 HM3D 物体搜索分别达到 92% 和 82%；人形全身控制基准中优于基线的任务为 30 项中的 13 项。论文同时指出推理时延高：一个 30 秒运动任务需 250 次模型调用，平均每次 39.86 秒，推理期间仿真暂停。
- **对 wiki 的映射：** [GPT 6 Astra 具身策略评测](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md) — 增加部署时延与跨任务能力边界的对照来源。

## 参考来源

- [arXiv 摘要与元数据](https://arxiv.org/abs/2609.38537)
