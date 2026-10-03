---
type: entity
tags: [paper, humanoid, whole-body-control, teleoperation, terrain-adaptation, perception]
status: complete
updated: 2026-10-03
arxiv: "2609.39000"
related:
  - ./nexus-humanoid.md
  - ../tasks/teleoperation.md
  - ../tasks/humanoid-locomotion.md
  - ../concepts/whole-body-control.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md
summary: "NEXUS（arXiv:2609.39000）：把地形适配加入感知式全身遥操作，通过地形修正参考动作并以深度和本体反馈跟踪。"
---

# NEXUS：面向地形适配遥操作的感知式全身控制

**NEXUS**（*Perceptive Whole-Body Control for Terrain-Adaptive Teleoperation*，[arXiv:2609.39000](https://arxiv.org/abs/2609.39000)，[项目页](https://nexus-humanoid.github.io/)）研究人体动作参考如何适配机器人所处地形，并由全身策略执行。

## 一句话理解

先按地形调整要跟踪的人体动作，再用机载感知和本体反馈完成全身控制。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| WBC | Whole-Body Control | 全身控制 |
| G1 | Unitree G1 | 文章中的人形机器人平台 |

## 方法要点

- 离线处理参考动作，使接触目标与机器人前方地形相适应。
- 在线策略根据深度和身体历史跟踪修正后的参考。
- Day 2 来源报告其在 Unitree G1 上展示楼梯与斜坡遥操作。

## 与同名工作区分

本页对应 **arXiv:2609.39000** 的 *Perceptive Whole-Body Control for Terrain-Adaptive Teleoperation*。仓库中的 [较早 NEXUS 研究预告页](./nexus-humanoid.md)记录的是 *A Perceptive Foundation Policy for Cross-Domain Whole-Body Teleoperation*（2026-09-06 状态）；两者标题、时间与研究材料不同，不能合并。

## 参考来源

- [arXiv:2609.39000](https://arxiv.org/abs/2609.39000)
- [项目页](https://nexus-humanoid.github.io/)
- [Day 2 文章来源索引](../../sources/blogs/humanoid_motion_intelligence_day2_locomotion_motion_priors_2026_10_03.md)

## 评测

文章报告仿真控制成功率 91.072%，去除深度后为 87.771%；真机展示楼梯与斜坡。

## 与其他工作对比

与 [早期 NEXUS 预告](./nexus-humanoid.md) 是两项不同工作；本页对应 arXiv:2609.39000，早期页对应另一题目。

## 结论

NEXUS先按地形修正参考，再由全身策略执行；它与同名预告页是不同工作。

## 关联页面

- [nexus-humanoid](./nexus-humanoid.md)
- [teleoperation](../tasks/teleoperation.md)
- [whole-body-control](../concepts/whole-body-control.md)
