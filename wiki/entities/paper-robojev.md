---
type: entity
project_id: robojev
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-10-28
arxiv: "2610.33718"
project: https://huggingface.co/josefchen/RoboJev-27B
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-10-28-59-papers-technology-map.md
sources:
  - ../../sources/papers/robojev_arxiv_2610_33718.md
  - ../../sources/sites/robojev.md
  - ../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md
summary: "RoboJev（arXiv:2610.33718）：以任务指令、操作视频抽帧和可选辅助视角判断执行成功，用低秩适配训练视觉语言裁判。"
---

# RoboJev（arXiv:2610.33718）

**RoboJev**（*RoboJev: An Open Vision-Language Judge for Robot Success Detection*，[arXiv:2610.33718](https://arxiv.org/abs/2610.33718)，[项目页](https://huggingface.co/josefchen/RoboJev-27B)）来自 [具身智能小站 2026-10-28 盘点](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)，属「评测与基准」组。

## 一句话定义

**以任务指令、操作视频抽帧和可选辅助视角判断执行成功，用低秩适配训练视觉语言裁判。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RoboJev | RoboJev: An Open Vision-Language Judge for Robot Success Detection | 本文方法 / 基准的简称 |

## 为什么重要

- 判断机器人是否完成任务。提供成功检测研究入口，使用时仍需确认新场景的标签定义、失败类型与泛化。
- 开源结论：**已开源**（步骤 2.5，2026-10-28）。公众号只给 Hugging Face 模型页；页面链接的 GitHub 仓库 josefchen/robojev 经 git ls-remote 可达。
- 同期同组工作见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.33718](https://arxiv.org/abs/2610.33718) |
| **开源** | **已开源** |
| **要点** | 以任务指令、操作视频抽帧和可选辅助视角判断执行成功，用低秩适配训练视觉语言裁判。 |
| **文内指标** | 公众号转述：提供成功检测研究入口，使用时仍需确认新场景的标签定义、失败类型与泛化。 |

## 源码运行时序图

**不适用（已开源）** — 公众号只给 Hugging Face 模型页；页面链接的 GitHub 仓库 josefchen/robojev 经 git ls-remote 可达。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号转述：提供成功检测研究入口，使用时仍需确认新场景的标签定义、失败类型与泛化。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**判断机器人是否完成任务：提供成功检测研究入口，使用时仍需确认新场景的标签定义、失败类型与泛化。**

1. 开源边界：**已开源** — 公众号只给 Hugging Face 模型页；页面链接的 GitHub 仓库 josefchen/robojev 经 git ls-remote 可达。
2. 核心机制：以任务指令、操作视频抽帧和可选辅助视角判断执行成功，用低秩适配训练视觉语言裁判。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-10-28-59-papers-technology-map](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)

## 参考来源

- [robojev_arxiv_2610_33718.md](../../sources/papers/robojev_arxiv_2610_33718.md)
- [wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)
- [arXiv:2610.33718](https://arxiv.org/abs/2610.33718)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.33718)
- [项目页](https://huggingface.co/josefchen/RoboJev-27B)
