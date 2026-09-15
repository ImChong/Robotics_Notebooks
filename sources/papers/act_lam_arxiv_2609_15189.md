# Reconstructing Is Not Acting: Action-Centric Latent Dynamics Modeling

> 来源归档（ingest）

- **标题：** Reconstructing Is Not Acting: Action-Centric Latent Dynamics Modeling
- **简称：** ACT-LAM
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.15189>
- **PDF：** <https://arxiv.org/pdf/2609.15189>
- **代码：** <https://github.com/DingjieFu/ACT-LAM>

- **入库日期：** 2026-09-15
- **索引来源：** [具身智能小站 9+EffVLA 盘点](../blogs/wechat_embodied_station_9_papers_resources_effvla_2026-09-15.md)
- **一句话说明：** AQ-IDM + AT-FDM 拆解动作提取与利用；VP² 聚合成功率 49.04%（+7.6%），约 55M 可训练参数。

## 开源状态（步骤 2.5，2026-09-15）

**结论：已开源**

## 核心摘录

### 摘录 1

AQ-IDM + AT-FDM 拆解动作提取与利用；VP² 聚合成功率 49.04%（+7.6%），约 55M 可训练参数。

**对 wiki 的映射：** [paper-act-lam](../../wiki/entities/paper-act-lam.md)

### 摘录 2（官方 abstract 要点，2026-09-15 补录）

- **问题命名：** 论文提出 **reconstruction–action mismatch** — 潜动作模型（LAM）的重建误差更低 **并不必然** 带来更好的潜动力学或下游表现。
- **归因两处欠约束：**
  1. **IDM** 没有被显式要求把 **动作相关转移** 与 **无关外观变化（nuisance appearance）** 区分开；
  2. **FDM** 可以靠 **当前状态的预测捷径** 绕开推断出的潜动作，从而低估它。
- **AQ-IDM：** 可学习 **action query** + **门控聚合**，在 **不施加强信息瓶颈** 的前提下选择性抽取动作相关转移线索。
- **AT-FDM：** 把潜动作投影成 **action token**，与不断演化的状态表示 **逐步交互**，实现持续的 state-aware 动作条件化。
- **评测与结果：** 在多个机器人数据集与 **VP²** 基准上报告更强的潜动作一致性、前向动力学与下游视觉规划；**VP² 聚合成功率较此前最佳 +7.6%**，且 **可训练参数更少、计算开销更低**。

**对 wiki 的映射：** 同上（补入该页「实验与评测」「与其他工作对比」两节）

## 当前提炼状态

- [x] 项目页/仓库核查
- [x] wiki 映射
