---
type: entity
project_id: ttco-test-time-compute-humanoid-bfm
tags:
  - paper
  - behavior-foundation-model
  - humanoid
  - test-time-optimization
status: complete
updated: 2026-10-14
arxiv: "2610.14896"
project: https://bfmttc.github.io/
related:
  - ../concepts/behavior-foundation-model.md
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-10-14-41-papers-technology-map.md
sources:
  - ../../sources/papers/ttco_test_time_compute_humanoid_bfm_arxiv_2610_14896.md
  - ../../sources/sites/ttco-test-time-compute-humanoid-bfm.md
  - ../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md
summary: "TTCO（arXiv:2610.14896）：闭环仿真中搜索更合适的输入命令，让既有行为基础模型更接近期望运动，并可生成模型专属动作标签；无需重训。"
---

# TTCO（arXiv:2610.14896）

**TTCO**（*Test-Time Compute for Humanoid Behavior Foundation Models*，[arXiv:2610.14896](https://arxiv.org/abs/2610.14896)，[项目页](https://bfmttc.github.io/)）来自 [具身智能小站 2026-10-14 盘点](../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)，属「人形与移动操作」组。

## 一句话定义

**闭环仿真中搜索更合适的输入命令，让既有行为基础模型更接近期望运动，并可生成模型专属动作标签；无需重训。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BFM | Behavior Foundation Model | 行为基础模型 |
| TTC | Test-Time Compute | 测试时计算 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- BFM 的命令–执行差距常由命令而非模型造成。
- 开源结论：**待发布**（步骤 2.5，2026-10-14）。项目页列出 GitHub 链接（Gamma-Robot/bfm_ttc），但入库日 ls-remote 不可达，按待发布处理。
- 同期同组工作见 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.14896](https://arxiv.org/abs/2610.14896) |
| **开源** | **待发布** |
| **要点** | 依赖仿真与采样计算。 |
| **文内指标** | 以原文为准。 |

## 源码运行时序图

**不适用（待发布）** — 项目页列出 GitHub 链接（Gamma-Robot/bfm_ttc），但入库日 ls-remote 不可达，按待发布处理。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 以原文为准。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**先用测试时优化压缩命令执行差距，再决定是否重训 BFM。**

1. 开源边界：**待发布** — 项目页列出 GitHub 链接（Gamma-Robot/bfm_ttc），但入库日 ls-remote 不可达，按待发布处理。
2. 核心机制：依赖仿真与采样计算。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [behavior-foundation-model](../concepts/behavior-foundation-model.md)
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-10-14-41-papers-technology-map](../overview/embodied-station-2026-10-14-41-papers-technology-map.md)

## 参考来源

- [ttco_test_time_compute_humanoid_bfm_arxiv_2610_14896.md](../../sources/papers/ttco_test_time_compute_humanoid_bfm_arxiv_2610_14896.md)
- [wechat_embodied_station_41_papers_streamingwam_2026-10-14.md](../../sources/blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)
- [arXiv:2610.14896](https://arxiv.org/abs/2610.14896)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.14896)
- [项目页](https://bfmttc.github.io/)
