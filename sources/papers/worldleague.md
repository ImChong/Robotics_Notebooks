# Benchmarking Physical Agency (WorldLeague)｜具身系统物理目标达成能力评测

- **类型：** paper
- **作者：** Xinyang Zhou、Qi Li、Xinchao Wang
- **机构：** 新加坡国立大学 xML Lab
- **版本：** arXiv:2611.30989v1（2026-11-23）
- **原文：** <https://arxiv.org/html/2611.30989v1> · [PDF](https://arxiv.org/pdf/2611.30989)
- **项目页：** [官方项目](../sites/worldleague.md)
- **代码：** [官方代码](../repos/worldleague.md)
- **核对范围：** 2026-11-25 查阅 HTML 原文关键机制、结果及讨论摘录，项目主页和 README；全文与必要附录尚未逐段核完。
- **通读：** 未完成

## 核心摘录（整理）

- 60 项任务分为 WorldSeek、WorldShape、WorldShift，覆盖导航、定位、几何、物理、记忆、控制。VLM 经 WorldHarness 选择动作，连续策略使用自身接口；从物理轨迹计算目标项完成度及全部条件满足率，终端 QA 单列，不代替物理分数。
- 13 个系统中最高平均 Interactive Score 28.58/100，完成 13/60；347 个终端 QA 全对的 VLM 回合中 249 个物理分数为零（71.8%）。连续控制评估覆盖 59 项，跨系统比较要同时注明接口、任务覆盖和预算。
- 仅仿真，不代表真机成功率。推理期间物理环境继续运行，推理时间和接口抽象影响比较。官方代码和基准数据已列出；场景生成公开入口在 README 中仍为 TODO。

对 wiki 的映射：[具身系统物理目标达成能力评测](../../wiki/entities/paper-worldleague.md)。
