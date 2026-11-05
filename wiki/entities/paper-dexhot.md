---
type: entity
project_id: dexhot
tags:
  - paper
  - dexterous
  - imitation-learning
status: complete
updated: 2026-11-05
arxiv: "2611.05119"
project: https://dex-hot.github.io/
related:
  - ../methods/imitation-learning.md
  - ../concepts/dexterous-kinematics.md
  - ../tasks/manipulation.md
  - ../concepts/motion-retargeting.md
  - ../overview/embodied-station-2026-11-05-43-papers-technology-map.md
sources:
  - ../../sources/papers/dexhot_arxiv_2611_05119.md
  - ../../sources/sites/dexhot.md
  - ../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md
summary: "DexHOT（arXiv:2611.05119）：下一步能否接着转动物体，取决于手指落在哪里。"
---

# DexHOT（arXiv:2611.05119）

**DexHOT**（*DexHOT: Learning Hand-Object Configuration Transfer for Dexterous Manipulation*，[arXiv:2611.05119](https://arxiv.org/abs/2611.05119)，[项目页](https://dex-hot.github.io/)）来自 [具身智能小站 2026-11-05 盘点](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)，属「灵巧操作与示范学习」组。

## 一句话定义

**下一步能否接着转动物体，取决于手指落在哪里。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DexHOT | DexHOT: Learning Hand-Object Configuration Transfer for Dexterous Manipulation | 本文方法 / 基准的简称 |

## 为什么重要

- 给灵巧手同时指定物体朝向和手指姿势。DexHOT 把手形和物体姿态一起设为目标，学会短距离调整再串联动作，真机上 200 次姿态转换成功了 123 次。
- 开源结论：**部分开源**（步骤 2.5，2026-11-05）。公众号未给代码链接；项目页链接的 dex-hot/DexHOT-Code 经 git ls-remote 可达，是否即论文完整代码未核对。
- 同期同组工作见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.05119](https://arxiv.org/abs/2611.05119) |
| **开源** | **部分开源** |
| **要点** | 下一步能否接着转动物体，取决于手指落在哪里。 |
| **文内指标** | 公众号转述：DexHOT 把手形和物体姿态一起设为目标，学会短距离调整再串联动作，真机上 200 次姿态转换成功了 123 次。 |

## 源码运行时序图

**部分开源** — 公众号未给代码链接；项目页链接的 dex-hot/DexHOT-Code 经 git ls-remote 可达，是否即论文完整代码未核对。 待可运行代码公开后补图。

## 实验与评测

- 公众号转述：DexHOT 把手形和物体姿态一起设为目标，学会短距离调整再串联动作，真机上 200 次姿态转换成功了 123 次。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**给灵巧手同时指定物体朝向和手指姿势：DexHOT 把手形和物体姿态一起设为目标，学会短距离调整再串联动作，真机上 200 次姿态转换成功了 123 次。**

1. 开源边界：**部分开源** — 公众号未给代码链接；项目页链接的 dex-hot/DexHOT-Code 经 git ls-remote 可达，是否即论文完整代码未核对。
2. 核心机制：下一步能否接着转动物体，取决于手指落在哪里。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [imitation-learning](../methods/imitation-learning.md)
- [dexterous-kinematics](../concepts/dexterous-kinematics.md)
- [manipulation](../tasks/manipulation.md)
- [motion-retargeting](../concepts/motion-retargeting.md)
- [embodied-station-2026-11-05-43-papers-technology-map](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)

## 参考来源

- [dexhot_arxiv_2611_05119.md](../../sources/papers/dexhot_arxiv_2611_05119.md)
- [wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)
- [arXiv:2611.05119](https://arxiv.org/abs/2611.05119)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.05119)
- [项目页](https://dex-hot.github.io/)
