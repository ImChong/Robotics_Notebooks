---
type: entity
project_id: adadextrack
tags:
  - paper
  - manipulation
  - contact-rich
status: complete
updated: 2026-10-28
arxiv: "2610.33852"
project: https://janebek.github.io/AdaDexTrack/
code: https://github.com/Janebek/AdaDexTrack
related:
  - ../tasks/manipulation.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/dexterous-kinematics.md
  - ../concepts/motion-retargeting.md
  - ../overview/embodied-station-2026-10-28-59-papers-technology-map.md
sources:
  - ../../sources/papers/adadextrack_arxiv_2610_33852.md
  - ../../sources/repos/adadextrack.md
  - ../../sources/sites/adadextrack.md
  - ../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md
summary: "AdaDexTrack（arXiv:2610.33852）：先蒸馏通用灵巧追踪器，再用闭环调制器持续调整参考轨迹、物体表征和位置目标。"
---

# AdaDexTrack（arXiv:2610.33852）

**AdaDexTrack**（*AdaDexTrack: Dynamic Modulation for Adaptive and Generalizable Dexterous Manipulation Tracking*，[arXiv:2610.33852](https://arxiv.org/abs/2610.33852)，[项目页](https://janebek.github.io/AdaDexTrack/)，[代码](https://github.com/Janebek/AdaDexTrack)）来自 [具身智能小站 2026-10-28 盘点](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)，属「灵巧与接触操作」组。

## 一句话定义

**先蒸馏通用灵巧追踪器，再用闭环调制器持续调整参考轨迹、物体表征和位置目标。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AdaDexTrack | AdaDexTrack: Dynamic Modulation for Adaptive and Generalizable Dexterous Manipulation Tracking | 本文方法 / 基准的简称 |

## 为什么重要

- 让灵巧手在追踪中持续修正参考。在未见物体、轨迹及真机迁移中考察鲁棒性，核心是避免固定参考让接触误差逐步累积。
- 开源结论：**已开源**（步骤 2.5，2026-10-28）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.33852](https://arxiv.org/abs/2610.33852) |
| **开源** | **已开源** |
| **要点** | 先蒸馏通用灵巧追踪器，再用闭环调制器持续调整参考轨迹、物体表征和位置目标。 |
| **文内指标** | 公众号转述：在未见物体、轨迹及真机迁移中考察鲁棒性，核心是避免固定参考让接触误差逐步累积。 |

## 源码运行时序图

以下为按仓库入口（`Janebek/AdaDexTrack`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as Janebek/AdaDexTrack
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 AdaDexTrack 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号转述：在未见物体、轨迹及真机迁移中考察鲁棒性，核心是避免固定参考让接触误差逐步累积。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**让灵巧手在追踪中持续修正参考：在未见物体、轨迹及真机迁移中考察鲁棒性，核心是避免固定参考让接触误差逐步累积。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：先蒸馏通用灵巧追踪器，再用闭环调制器持续调整参考轨迹、物体表征和位置目标。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [contact-rich-manipulation](../concepts/contact-rich-manipulation.md)
- [dexterous-kinematics](../concepts/dexterous-kinematics.md)
- [motion-retargeting](../concepts/motion-retargeting.md)
- [embodied-station-2026-10-28-59-papers-technology-map](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)

## 参考来源

- [adadextrack_arxiv_2610_33852.md](../../sources/papers/adadextrack_arxiv_2610_33852.md)
- [wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)
- [arXiv:2610.33852](https://arxiv.org/abs/2610.33852)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.33852)
- [项目页](https://janebek.github.io/AdaDexTrack/)
- [https://github.com/Janebek/AdaDexTrack](https://github.com/Janebek/AdaDexTrack)
