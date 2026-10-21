---
type: entity
project_id: touch2robot
tags:
  - paper
  - teleoperation
  - tactile
  - demonstration
  - retargeting
status: complete
updated: 2026-10-18
arxiv: "2609.24660"
project: https://touch2robot.github.io/
related:
  - ../tasks/teleoperation.md
  - ../concepts/tactile-sensing.md
  - ../concepts/motion-retargeting.md
  - ../overview/embodied-station-2026-10-18-26-papers-technology-map.md
sources:
  - ../../sources/papers/touch2robot_arxiv_2609_24660.md
  - ../../sources/sites/touch2robot.md
  - ../../sources/blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md
summary: "Touch2Robot（arXiv:2609.24660）：把手套触觉、人体动作和物体跟踪用于实时重定向，让示范者在采集时看到目标机器人的接触反馈。"
---

# Touch2Robot（arXiv:2609.24660）

**Touch2Robot**（*Touch2Robot: Robot Touch in the Human Demonstration Loop*，[arXiv:2609.24660](https://arxiv.org/abs/2609.24660)，[项目页](https://touch2robot.github.io/)）来自 [具身智能小站 2026-10-18 盘点](../../sources/blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)，属「数据生成与示范」组。

## 一句话定义

**把手套触觉、人体动作和物体跟踪用于实时重定向，让示范者在采集时看到目标机器人的接触反馈。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Retargeting | Motion Retargeting | 跨形态动作重定向 |

## 为什么重要

- 人与机器人形态差异要在采集阶段就纠正，事后重定向难以补回接触信息。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-10-18）。项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。
- 同期同组工作见 [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../overview/embodied-station-2026-10-18-26-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.24660](https://arxiv.org/abs/2609.24660) |
| **开源** | **项目页未列 GitHub** |
| **要点** | 手套触觉 + 人体动作 + 物体跟踪 → 实时重定向，并把目标机器人的接触反馈回显给示范者。 |
| **文内指标** | 公众号称采集阶段纠正形态差异改善真机回放；任务完成度不能混称成功率。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号称采集阶段纠正形态差异改善真机回放；任务完成度不能混称成功率。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../overview/embodied-station-2026-10-18-26-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**示范采集回路里加入机器人侧接触反馈，是比事后重定向更早的形态对齐手段。**

1. 开源边界：**项目页未列 GitHub** — 项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。
2. 核心机制：手套触觉 + 人体动作 + 物体跟踪 → 实时重定向，并把目标机器人的接触反馈回显给示范者。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [teleoperation](../tasks/teleoperation.md)
- [tactile-sensing](../concepts/tactile-sensing.md)
- [motion-retargeting](../concepts/motion-retargeting.md)
- [embodied-station-2026-10-18-26-papers-technology-map](../overview/embodied-station-2026-10-18-26-papers-technology-map.md)

## 参考来源

- [touch2robot_arxiv_2609_24660.md](../../sources/papers/touch2robot_arxiv_2609_24660.md)
- [wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md](../../sources/blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)
- [arXiv:2609.24660](https://arxiv.org/abs/2609.24660)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.24660)
- [项目页](https://touch2robot.github.io/)
