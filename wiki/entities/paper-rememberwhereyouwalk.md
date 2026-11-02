---
type: entity
project_id: rememberwhereyouwalk
tags:
  - paper
  - navigation
  - mobile-robot
status: complete
updated: 2026-10-28
arxiv: "2610.37203"
project: https://leonmgns.github.io/RememberWhereYouWalkWebiste/
code: https://github.com/LeonMGNS/RememberWhereYouWalk
related:
  - ../tasks/locomotion.md
  - ../concepts/sim2real.md
  - ../methods/reinforcement-learning.md
  - ../tasks/vision-language-navigation.md
  - ../overview/embodied-station-2026-10-28-59-papers-technology-map.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md
  - ../../sources/papers/rememberwhereyouwalk_arxiv_2610_37203.md
  - ../../sources/repos/rememberwhereyouwalk.md
  - ../../sources/sites/rememberwhereyouwalk.md
  - ../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md
summary: "RememberWhereYouWalk（arXiv:2610.37203）：以位置索引的局部空间矩阵记忆替代单纯循环状态，在迷宫探索中保留离开视野的布局信息。"
---

# RememberWhereYouWalk（arXiv:2610.37203）

**RememberWhereYouWalk**（*Remember Where You Walk: Learning Spatial Memory for End-to-End Legged Robot Exploration*，[arXiv:2610.37203](https://arxiv.org/abs/2610.37203)，[项目页](https://leonmgns.github.io/RememberWhereYouWalkWebiste/)，[代码](https://github.com/LeonMGNS/RememberWhereYouWalk)）来自 [具身智能小站 2026-10-28 盘点](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)，属「导航、无人机与群体」组。

## 一句话定义

**以位置索引的局部空间矩阵记忆替代单纯循环状态，在迷宫探索中保留离开视野的布局信息。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RememberWhereYouWalk | Remember Where You Walk: Learning Spatial Memory for End-to-End Legged Robot Exploration | 本文方法 / 基准的简称 |

## 为什么重要

- 把走过的地方存入空间记忆。探索覆盖率与强基线相近，价值主要体现为更好的空间重建与记忆。
- 开源结论：**已开源**（步骤 2.5，2026-10-28）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.37203](https://arxiv.org/abs/2610.37203) |
| **开源** | **已开源** |
| **要点** | 以位置索引的局部空间矩阵记忆替代单纯循环状态，在迷宫探索中保留离开视野的布局信息。 |
| **文内指标** | 公众号转述：探索覆盖率与强基线相近，价值主要体现为更好的空间重建与记忆。 |

## 源码运行时序图

以下为按仓库入口（`LeonMGNS/RememberWhereYouWalk`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as LeonMGNS/RememberWhereYouWalk
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 RememberWhereYouWalk 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：探索覆盖率与强基线相近，价值主要体现为更好的空间重建与记忆。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**把走过的地方存入空间记忆：探索覆盖率与强基线相近，价值主要体现为更好的空间重建与记忆。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：以位置索引的局部空间矩阵记忆替代单纯循环状态，在迷宫探索中保留离开视野的布局信息。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [sim2real](../concepts/sim2real.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [embodied-station-2026-10-28-59-papers-technology-map](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)

## 参考来源

- [rememberwhereyouwalk_arxiv_2610_37203.md](../../sources/papers/rememberwhereyouwalk_arxiv_2610_37203.md)
- [wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)
- [arXiv:2610.37203](https://arxiv.org/abs/2610.37203)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.37203)
- [项目页](https://leonmgns.github.io/RememberWhereYouWalkWebiste/)
- [https://github.com/LeonMGNS/RememberWhereYouWalk](https://github.com/LeonMGNS/RememberWhereYouWalk)
