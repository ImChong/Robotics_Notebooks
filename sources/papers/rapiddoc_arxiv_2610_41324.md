# RapidDOC（arXiv:2610.41324）

> 来源归档（paper）

- **标题：** RapidDOC: Riccati-Preconditioned Implicit Differentiation for Differentiable Optimal Control under Operator Mismatch
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.41324>
- **PDF：** <https://arxiv.org/pdf/2610.41324>
- **代码：** <https://github.com/opt-man/RapidDOC>
- **入库日期：** 2026-11-03
- **线索来源：** [机器人不仅要完成任务，还要知道哪里出了错](../blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)（具身智能小站，2026-11-03）
- **一句话说明：** RapidDOC 将前向分解缓存用于隐式微分的预条件，检查目标残差后再迭代修正，避免算子不匹配时直接复用产生错误梯度，适合可微最优控制系统。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-03）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 复用求解分解，也要修正梯度。
- RapidDOC 将前向分解缓存用于隐式微分的预条件，检查目标残差后再迭代修正，避免算子不匹配时直接复用产生错误梯度，适合可微最优控制系统。
- 指标：公众号未给具体数字，本页不引用。

## 对 wiki 的映射

- [paper-rapiddoc](../../wiki/entities/paper-rapiddoc.md)
- [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-03-49-papers-technology-map.md)
