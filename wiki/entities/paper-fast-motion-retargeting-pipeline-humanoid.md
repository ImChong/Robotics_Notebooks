---
type: entity
tags:
  - paper
  - humanoid
status: complete
updated: 2026-10-31
arxiv: "2610.28470"
project_id: fast-motion-retargeting-pipeline-humanoid
project: https://daroslab.github.io/frdv/
related:
  - ../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-10-31-50-papers-technology-map.md
sources:
  - ../../sources/papers/fast-motion-retargeting-pipeline-humanoid_arxiv_2610_28470.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md
  - ../../sources/blogs/wechat_embodied_station_50_papers_scenerig_photo_to_sim_2026-10-31.md
summary: "Fast Retargeting：面向规模化人形学习的快速动作重定向流程（arXiv:2610.28470）：结合定制 SQP、实时迭代和分块并行，将速度提升至最高每秒 9156 帧；再用接触隐式 iLQR 筛除或修正动力学不可执行、悬脚和滑脚动作。"
---

# Fast Retargeting：面向规模化人形学习的快速动作重定向流程

**Fast Retargeting**（*Fast Motion Retargeting Pipeline for Scalable Humanoid Robot Learning*，[arXiv:2610.28470](https://arxiv.org/abs/2610.28470)）收录于 senlanke「具身运控lab」10-19–10-23 前沿论文动态（人形/四足）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**Fast Retargeting：面向规模化人形学习的快速动作重定向流程——结合定制 SQP、实时迭代和分块并行，将速度提升至最高每秒 9156 帧；再用接触隐式 iLQR 筛除或修正动力学不可执行、悬脚和滑脚动作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 周报所述待解决的问题：将大型人体动作库转移到不同人形机器人时，传统轨迹优化速度慢，还需要逐动作人工修复。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.28470](https://arxiv.org/abs/2610.28470) · [PDF](https://arxiv.org/pdf/2610.28470) |
| **作者（arXiv）** | Daniel Marew, Sangjun Park, Nisal Perera, Donghyun Kim |
| **机构（周报所述，未逐条核实）** | 马萨诸塞大学阿默斯特分校 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页/评论给出项目页或外链（https://daroslab.github.io/frdv/）；未发现代码仓库链接（2026-10-26） |
| **周报所述创新点** | 结合定制 SQP、实时迭代和分块并行，将速度提升至最高每秒 9156 帧；再用接触隐式 iLQR 筛除或修正动力学不可执行、悬脚和滑脚动作。 |

### arXiv 摘要节选（英文原文）

> While large-scale human motion capture datasets provide a rich source of demonstrations for humanoid robot learning, retargeting the diverse motions to different humanoid robots remains a nontrivial challenge. Existing kinodynamic trajectory optimization pipelines are computationally expensive and often require manual adjustments per motion.

## 源码运行时序图

**不适用**：arXiv 摘要页/评论给出项目页或外链（https://daroslab.github.io/frdv/）；未发现代码仓库链接，无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [人形/四足周报技术地图](../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页/评论给出项目页或外链（https://daroslab.github.io/frdv/）；未发现代码仓库链接 |

## 结论

**Fast Retargeting：面向规模化人形学习的快速动作重定向流程。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map](../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md)
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [fast-motion-retargeting-pipeline-humanoid_arxiv_2610_28470.md](../../sources/papers/fast-motion-retargeting-pipeline-humanoid_arxiv_2610_28470.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
- [arXiv:2610.28470](https://arxiv.org/abs/2610.28470)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.28470)
