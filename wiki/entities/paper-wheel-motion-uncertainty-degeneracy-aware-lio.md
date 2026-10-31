---
type: entity
tags:
  - paper
  - legged-robot
  - locomotion
status: complete
updated: 2026-10-31
arxiv: "2610.26550"
project_id: wheel-motion-uncertainty-degeneracy-aware-lio
project: https://weihairuo.github.io/wheel-motion-uncertainty/
related:
  - ../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md
  - ../tasks/locomotion.md
  - ../overview/embodied-station-2026-10-31-50-papers-technology-map.md
sources:
  - ../../sources/papers/wheel-motion-uncertainty-degeneracy-aware-lio_arxiv_2610_26550.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md
  - ../../sources/blogs/wechat_embodied_station_50_papers_scenerig_photo_to_sim_2026-10-31.md
summary: "Wheel Uncertainty LIO：面向四足轮腿机器人的轮运动不确定性学习与退化感知激光—惯性里程计（arXiv:2610.26550）：从本体历史为每个轮子学习方向相关过程协方差，在可微滤波器中训练；同时抑制激光雷达几何弱方向的约束，使轮运动和激光约束根据各自退化程度自适应加权。"
---

# Wheel Uncertainty LIO：面向四足轮腿机器人的轮运动不确定性学习与退化感知激光—惯性里程计

**Wheel Uncertainty LIO**（*Learning Wheel Motion Uncertainty for Degeneracy-Aware LiDAR-Inertial Odometry on Quadrupedal Wheel-Legged Robots*，[arXiv:2610.26550](https://arxiv.org/abs/2610.26550)）收录于 senlanke「具身运控lab」10-19–10-23 前沿论文动态（人形/四足）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**Wheel Uncertainty LIO：面向四足轮腿机器人的轮运动不确定性学习与退化感知激光—惯性里程计——从本体历史为每个轮子学习方向相关过程协方差，在可微滤波器中训练；同时抑制激光雷达几何弱方向的约束，使轮运动和激光约束根据各自退化程度自适应加权。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 周报所述待解决的问题：走廊等几何退化环境需要轮运动提供额外约束，但打滑、悬空会使轮速预测失真，二值接触状态不能反映不同方向上的可靠性。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.26550](https://arxiv.org/abs/2610.26550) · [PDF](https://arxiv.org/pdf/2610.26550) |
| **作者（arXiv）** | Hairuo Wei, Yunfan Ren, Renda Yang, Qiangsheng Huang, Zhongyu Li |
| **机构（周报所述，未逐条核实）** | 香港中文大学、同济大学、仙工智能 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页/评论给出项目页或外链（https://weihairuo.github.io/wheel-motion-uncertainty/）；未发现代码仓库链接（2026-10-26） |
| **周报所述创新点** | 从本体历史为每个轮子学习方向相关过程协方差，在可微滤波器中训练；同时抑制激光雷达几何弱方向的约束，使轮运动和激光约束根据各自退化程度自适应加权。 |

### arXiv 摘要节选（英文原文）

> Wheel kinematics can support odometry for quadrupedal wheel-legged robots when LiDAR geometry provides weak motion constraints. However, slip and contact loss make rolling predictions unreliable, and contact state alone cannot determine their reliability across motion directions.

## 源码运行时序图

**不适用**：arXiv 摘要页/评论给出项目页或外链（https://weihairuo.github.io/wheel-motion-uncertainty/）；未发现代码仓库链接，无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [人形/四足周报技术地图](../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页/评论给出项目页或外链（https://weihairuo.github.io/wheel-motion-uncertainty/）；未发现代码仓库链接 |

## 结论

**Wheel Uncertainty LIO：面向四足轮腿机器人的轮运动不确定性学习与退化感知激光—惯性里程计。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map](../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md)
- [locomotion](../tasks/locomotion.md)

## 参考来源

- [wheel-motion-uncertainty-degeneracy-aware-lio_arxiv_2610_26550.md](../../sources/papers/wheel-motion-uncertainty-degeneracy-aware-lio_arxiv_2610_26550.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
- [arXiv:2610.26550](https://arxiv.org/abs/2610.26550)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.26550)
