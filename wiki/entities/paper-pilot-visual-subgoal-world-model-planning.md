---
type: entity
project_id: pilot-visual-subgoal-world-model-planning
tags:
  - paper
  - world-model
  - planning
  - cem
  - subgoal
  - manipulation
status: complete
updated: 2026-10-14
arxiv: "2610.14335"
project: https://pilot-planner.github.io/
related:
  - ../methods/generative-world-models.md
  - ../concepts/world-action-models.md
  - ../tasks/manipulation.md
  - ../methods/cma-es.md
  - ../overview/embodied-station-2026-10-14-41-papers-technology-map.md
sources:
  - ../../sources/papers/pilot_visual_subgoal_world_model_planning_arxiv_2610_14335.md
  - ../../sources/sites/pilot-visual-subgoal-world-model-planning.md
  - ../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md
summary: "PILOT (Visual Subgoals)（arXiv:2610.14335）：先生成中间视觉子目标，再用策略初始化 CEM 搜索，通过动作条件世界模型滚动优化。"
---

# PILOT (Visual Subgoals)（arXiv:2610.14335）

**PILOT (Visual Subgoals)**（*PILOT: Guiding Long-Horizon World-Model Planning for Robot Manipulation with Visual Subgoals and Policy Priors*，[arXiv:2610.14335](https://arxiv.org/abs/2610.14335)，[项目页](https://pilot-planner.github.io/)）来自 [具身智能小站 2026-10-14 盘点](../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)，属「世界模型与记忆」组。

## 一句话定义

**先生成中间视觉子目标，再用策略初始化 CEM 搜索，通过动作条件世界模型滚动优化。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CEM | Cross-Entropy Method | 交叉熵采样优化 |
| WM | World Model | 世界模型 |
| UR5e | Universal Robots UR5e | 机械臂平台 |

## 为什么重要

- 长程世界模型规划需要子目标与先验压缩搜索空间。
- 开源结论：**待发布**（步骤 2.5，2026-10-14）。项目页含「coming soon」字样，入库日未列 GitHub。与库内 PILOT 足式节点为不同论文。
- 同期同组工作见 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.14335](https://arxiv.org/abs/2610.14335) |
| **开源** | **待发布** |
| **要点** | UR5e 与仿真；需计入在线规划预算。 |
| **文内指标** | 以原文为准。 |

## 源码运行时序图

**不适用（待发布）** — 项目页含「coming soon」字样，入库日未列 GitHub。与库内 PILOT 足式节点为不同论文。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 以原文为准。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**视觉子目标 + 策略先验是让 WM 规划在长程任务可用的两个杠杆。**

1. 开源边界：**待发布** — 项目页含「coming soon」字样，入库日未列 GitHub。与库内 PILOT 足式节点为不同论文。
2. 核心机制：UR5e 与仿真；需计入在线规划预算。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [generative-world-models](../methods/generative-world-models.md)
- [world-action-models](../concepts/world-action-models.md)
- [manipulation](../tasks/manipulation.md)
- [cma-es](../methods/cma-es.md)
- [embodied-station-2026-10-14-41-papers-technology-map](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)

## 参考来源

- [pilot_visual_subgoal_world_model_planning_arxiv_2610_14335.md](../../sources/papers/pilot_visual_subgoal_world_model_planning_arxiv_2610_14335.md)
- [wechat_embodied_station_41_papers_streamingwam_2026-10-14.md](../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)
- [arXiv:2610.14335](https://arxiv.org/abs/2610.14335)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.14335)
- [项目页](https://pilot-planner.github.io/)
