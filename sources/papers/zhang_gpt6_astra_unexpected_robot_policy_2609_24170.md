# An Unexpected Robot Policy: Early Evaluations of GPT-6 Astra on RoboDojo and Beyond

> 来源归档（arXiv 预印本）

- **标题：** An Unexpected Robot Policy: Early Evaluations of GPT-6 Astra on RoboDojo and Beyond
- **类型：** paper / evaluation
- **原始链接：** <https://arxiv.org/abs/2609.24170>
- **作者：** Wenbo Zhang、Kaixuan Wang、Yutao Ouyang、Xiaoyu Huang、Liyang Li、Kailun Su、Weiyang Jin、Wenhao Chai、Haotian Liang、Zhiyang Dou、Yue Chen、Tianxing Chen
- **提交日期：** 2026-09-21（v1）
- **入库日期：** 2026-10-20
- **一句话说明：** 在 RoboDojo 42 项任务上比较 GPT-6 Astra 等通用模型作为机器人操纵策略，并与公开策略对照；显示语义任务优势与精细控制短板并存。

## 核心摘录（面向 wiki 编译）

### 1) 问题设定

- **摘录要点：** 研究将语言模型直接作为机器人操纵策略（LLM as policy），不做任务专用微调；评估其是否能从视觉观察中生成控制动作。
- **对 wiki 的映射：** [GPT 6 Astra 具身策略评测](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md) — 与既有独立评测互补，需区分作者、协议与数据集。

### 2) 评测范围与主要结果

- **摘录要点：** 在 RoboDojo 全部 42 个任务上评估三种 LLM，并与 40 个公开策略比较。GPT-6 Astra 按每任务 50 回合协议完成 2,100 次试验，平均成功率 22.48%、Score 28.97，高于该比较中的公开条目。
- **对 wiki 的映射：** [RoboDojo](../../wiki/entities/robodojo.md) — 扩展其通用模型作策略的评测案例。

### 3) 能力边界

- **摘录要点：** Astra 对语义理解要求较高、但不依赖高精度控制的任务表现较好；对精确操作、动态控制和复杂双臂协调表现较弱。单次示范的上下文实验没有带来整体收益，部分交互轨迹显示模型能在扰动后于回合内修正。
- **对 wiki 的映射：** [GPT 6 Astra 具身策略评测](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md) — 对照任务覆盖、协议与局限，不应将两项研究的成功率直接横比。

## 参考来源

- [arXiv 摘要与元数据](https://arxiv.org/abs/2609.24170)
