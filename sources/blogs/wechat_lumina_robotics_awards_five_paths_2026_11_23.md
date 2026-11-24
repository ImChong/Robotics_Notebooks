# 2026 机器人顶会 Best Paper 盘点：机器人通用能力背后的五条技术路径

- **作者：** Lumina具身智能
- **日期：** 2026-11-23 18:20
- **链接：** <https://mp.weixin.qq.com/s/tUaMXFoKzvQ8HSbGcijKxA>
- **类型：** blog / 二手技术综述
- **核对范围：** 用户提供的完整文章文字（五方向、十二项研究及趋势讨论）；与仓库已有十二篇独立论文节点及其原文阅读归档交叉核对。未重新逐篇全文复读；奖项以会议或作者官方公告为准。
- **通读：** 已完成
- **版权：** 仅保留出处与自主提炼，不转载原文或配图。

## 核心观点

训练成本、示范适配、物理感知与控制、跨阶段状态维护、现场失败恢复是不同的工程问题；不能用一次展示成功替代连续运行指标。本文提出的五方向是二手作者的分类，并非会议官方 taxonomy。

## 十二篇论文索引

| 路径 | 会议（文内） | 论文详情 | 一手阅读归档 |
|---|---|---|---|
| 训练效率 | RSS 2026 | [FlashSAC](../../wiki/entities/paper-flashsac.md) | [原文归档](../papers/flashsac_arxiv_2604_04539.md) |
| 训练效率 | CoRL 2026 | [SuperMimic](../../wiki/entities/paper-supermimic.md) | [原文归档](../papers/supermimic_arxiv_2610_28480.md) |
| 训练效率 | RSS 2026 | [Muninn](../../wiki/entities/paper-muninn-trajectory-diffusion-acceleration.md) | [原文归档](../papers/muninn_arxiv_2605_09999.md) |
| 动作数据 | ICRA 2026 | [OmniRetarget](../../wiki/entities/paper-hrl-stack-03-omniretarget.md) | [原文归档](../papers/omniretarget_reading_arxiv_2509_26633_or3.md) |
| 物理交互 | RSS 2026 | [NeuralActuator](../../wiki/entities/paper-neuralactuator-neural-actuation-modeling.md) | [原文归档](../papers/neuralactuator_arxiv_2607_11734.md) |
| 物理交互 | IROS 2026 | [SteadyTray](../../wiki/entities/paper-notebook-steadytray.md) | [原文归档](../papers/steadytray_arxiv_2603_10306.md) |
| 物理交互 | ICRA 2026 | [Do You Know Where Your Camera Is?](../../wiki/entities/paper-know-your-camera.md) | [原文归档](../papers/know-your-camera_arxiv_2510_02268.md) |
| 长程任务 | ICRA 2026 | [SymSkill](../../wiki/entities/paper-symskill.md) | [原文归档](../papers/symskill_arxiv_2510_01661.md) |
| 长程任务 | IROS 2026 | [LT-Mem](../../wiki/entities/paper-lt-mem.md) | [原文归档](../papers/lt_mem_arxiv_2608_19059.md) |
| 长程任务 | CoRL 2026 | [RoboTTT](../../wiki/entities/paper-robottt-test-time-training-vla-context.md) | [原文归档](../papers/robottt_arxiv_2607_15275.md) |
| 真实任务 | IROS 2026 | [Picking Bins Empty](../../wiki/entities/paper-picking-bins-empty.md) | [原文归档](../papers/picking-bins-empty_arxiv_2608_28175.md) |
| 真实任务 | IROS 2026 | [VL-Nav](../../wiki/entities/paper-vl-nav.md) | [原文归档](../papers/vl-nav_arxiv_2502_00931.md) |

## 证据口径

- RoboTTT 的 87% 是相对完成分增益，不能当绝对成功率。
- SuperMimic 的 129,514 段是训练动作覆盖评测；不同硬件的计算比较应保留原协议。
- OmniRetarget 项目展示时长与论文各数据源时长口径并不完全相同，按原文归档查阅，不重复转述总数。
- VL-Nav 的机载平台不等于全部推理均在板端完成。
- 奖项名称不作为技术结论依据。RSS 官方公告、[ICRA 官方奖项候选页](https://2026.ieee-icra.org/awards/)及作者项目页可交叉查阅；候选不等于获奖。本次不重新登记未逐项核实的获奖类别。

## 对 wiki 的映射

- [五条能力路径对比](../../wiki/comparisons/robotics-2026-awards-five-capability-paths.md)
- [此前同一组十二篇论文索引](wechat_shenlan_four_robotics_conferences_12_awards_2026_11_17.md)
