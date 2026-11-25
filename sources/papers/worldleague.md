# Benchmarking Physical Agency (WorldLeague)｜具身系统物理目标达成能力评测

- **类型：** paper
- **作者：** Xinyang Zhou、Qi Li、Xinchao Wang
- **机构：** 新加坡国立大学 xML Lab
- **版本：** arXiv:2611.30989v1（2026-11-23）
- **原文：** <https://arxiv.org/html/2611.30989v1> · [PDF](https://arxiv.org/pdf/2611.30989)
- **项目页：** [官方项目](../sites/worldleague.md)
- **代码：** [官方代码](../repos/worldleague.md)
- **核对范围：** 2026-11-25 通读 arXiv v1 正文 §1–5 与附录 A–F，核对目标判定、不同动作接口、种子/预算、失败回合和场景变体构造；不代表本地复跑基准。
- **通读：** 已完成

## 核心摘录（整理）

- 60 项任务分为 WorldSeek、WorldShape、WorldShift，覆盖导航、定位、几何、物理、记忆、控制。VLM 经 WorldHarness 选择动作，连续策略使用自身接口；从物理轨迹计算目标项完成度及全部条件满足率，终端 QA 单列，不代替物理分数。
- 13 个系统中最高平均 Interactive Score 28.58/100，完成 13/60；347 个终端 QA 全对的 VLM 回合中 249 个物理分数为零（71.8%）。连续控制评估覆盖 59 项，跨系统比较要同时注明接口、任务覆盖和预算。
- 仅仿真，不代表真机成功率。推理期间物理环境继续运行，推理时间和接口抽象影响比较。官方代码和基准数据已列出；场景生成公开入口在 README 中仍为 TODO。

- 论文 §4.3 的 T41 例子可获 100 分目标项进展却因独立约束不满足而失败；附录 D.5 与 D.6 分别核对错误终止和隐藏反馈、动作抽象、记事功能的接口消融。

对 wiki 的映射：[具身系统物理目标达成能力评测](../../wiki/entities/paper-worldleague.md)。
