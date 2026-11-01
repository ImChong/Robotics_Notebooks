# PrismPlan（arXiv:2610.23884）

> 来源归档（paper）

- **标题：** PrismPlan: A Shared GPU Substrate for Motion Planning Across Paradigms
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.23884>
- **PDF：** <https://arxiv.org/pdf/2610.23884>
- **代码：** <https://github.com/XiaoyubotMotionTeam/PrismPlan>
- **入库日期：** 2026-11-01
- **线索来源：** [模型还在思考，目标已经移动：机器人评测该把时间算进来吗？](../blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)（具身智能小站，2026-11-01）
- **一句话说明：** PrismPlan把采样、搜索和优化接到同一GPU运动学与碰撞计算底座；统一预算下比较多种规划器，结果揭示维度和问题类型会改变算法取舍。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-01）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 规划器公平比较，先共用碰撞后端。
- PrismPlan把采样、搜索和优化接到同一GPU运动学与碰撞计算底座；统一预算下比较多种规划器，结果揭示维度和问题类型会改变算法取舍。
- 指标：公众号未给具体数字，本页不引用。

## 对 wiki 的映射

- [paper-prismplan](../../wiki/entities/paper-prismplan.md)
- [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-01-56-papers-technology-map.md)
