# Simulator Contact Artifacts Audit（arXiv:2610.41857）

> 来源归档（paper）

- **标题：** When the Simulator Fails, Not the Policy:Auditing Contact Artifactsin Simulated Robot Benchmarks
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.41857>
- **PDF：** <https://arxiv.org/pdf/2610.41857>
- **代码：** <https://github.com/ALEXander22568/simuguard>
- **入库日期：** 2026-11-03
- **线索来源：** [机器人不仅要完成任务，还要知道哪里出了错](../blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)（具身智能小站，2026-11-03）
- **一句话说明：** SimuGuard 记录物理步状态并重放异常，通过单参数干预定位接触碰撞错误。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-03）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 失败可能来自仿真接触伪影。它提醒评测者先检查仿真器造成的物体弹飞，再把全部失败归因于策略。
- SimuGuard 记录物理步状态并重放异常，通过单参数干预定位接触碰撞错误。
- 指标：公众号转述：它提醒评测者先检查仿真器造成的物体弹飞，再把全部失败归因于策略。

## 对 wiki 的映射

- [paper-simulator-contact-artifacts-audit](../../wiki/entities/paper-simulator-contact-artifacts-audit.md)
- [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-03-49-papers-technology-map.md)
