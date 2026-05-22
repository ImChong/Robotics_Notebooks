# WEM 官方项目页（ZGCA-HMI-Lab）

> 来源归档

- **标题：** World-Ego Modeling for Long-Horizon Evolution in Hybrid Embodied Tasks — Project Page
- **类型：** site / project-page
- **URL：** <https://zgca-hmi-lab.github.io/WEM/>
- **关联论文：** <https://arxiv.org/abs/2605.19957>
- **关联代码：** <https://github.com/ZGCA-HMI-Lab/WEM>
- **机构：** ZGCA-HMI-Lab（中关村学院相关团队语境）；作者单位含 CASIA、UCAS、SJTU、PKU 等（见论文）
- **入库日期：** 2026-05-22
- **一句话说明：** 论文 **WEM / World-Ego Modeling / HTEWorld** 的公开落地页：概念与框架图、模型总览、数据集统计、HTEWorld 定量表（WorldArena + 导航–操作专项）、定性 rollout 与 BibTeX。

## 页面结构归纳

1. **Highlights：** World-Ego Modeling 范式、WEM 模型（RCA + CP-MoE）、HTEWorld 基准与 SOTA 叙事。
2. **图示：** `static/images/framework.png`（预测 + CP-MoE 三档解耦）、`model.png`（多轮混合任务 world/ego 分支）、`dataset.png`（HTEWorld 规模与词汇）。
3. **定量（与论文一致）：**
   - **WorldArena / EWMScore：** WEM **61.48** vs WoW-7B **53.44**、Cosmos-Predict 2.5-2B **54.83**、14B **55.41**、PAN-style **58.40**（同训练数据微调对比）。
   - **导航–操作六项：** Rollout chunk-boundary dynamics、Late-prefix state alignment、Chunk instruction-step retrieval、Phase-matched motion profile、Cross-phase discriminative margin、Frontier phase-hop consistency — WEM 均领先所列基线。
4. **定性：** 多轮指令示例（如捡罐丢垃圾桶、开罐移动、搬垃圾桶等 **导航→操作** 交错序列）。
5. **BibTeX：** `@article{wem2026, ... arXiv:2605.19957}`

## 对 wiki 的映射

- 与 [`sources/papers/wem_arxiv_2605_19957.md`](../papers/wem_arxiv_2605_19957.md) 互为补充：论文摘录偏方法细节，本页偏 **结果表与演示索引**。
- 沉淀：[`wiki/entities/paper-wem-world-ego-modeling.md`](../../wiki/entities/paper-wem-world-ego-modeling.md)
