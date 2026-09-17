# FIERCE（arXiv:2609.18651）

> 来源归档（paper）

- **标题：** FIERCE: From Generalist Robot Policies to Fast Specialists via Progress–Failure Feedback
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.18651>
- **PDF：** <https://arxiv.org/pdf/2609.18651>
- **代码：** <https://github.com/ar-mine/FIERCE>
- **入库日期：** 2026-09-17
- **一句话说明：** 用观测到的任务进展与动作条件失败风险共同塑造 RL 反馈，把通用策略蒸馏为低延迟专才；仿真 + 插销入孔 + 叠杯评测。

## 开源状态

- **部分开源**（步骤 2.5 核查，2026-09-17）：GitHub 仓已建，README 称 implementation 仍在准备中。

## 核心摘录

通用 VLA/策略提供初始化，重复插入/对齐/放置需要低延迟专才。FIERCE 联合 **progress** 与 **failure-risk** 反馈做 RL，产出紧凑 specialist。

**对 wiki 的映射**

- [paper-fierce](../../wiki/entities/paper-fierce.md)
- [9 篇技术地图](../../wiki/overview/perception-action-transfer-9-papers-technology-map.md)
