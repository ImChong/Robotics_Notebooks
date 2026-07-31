---
type: entity
tags: [paper, loco-manipulation, loco-manip-161-survey, humanoid]
status: complete
updated: 2026-07-16
venue: curated
summary: "EgoHumanoid 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹、遥操作/外骨骼数据编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型预测地形/场景表征。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。"
related:
  - ../overview/humanoid-loco-manip-161-papers-technology-map.md
  - ../overview/loco-manip-161-category-03-visuomotor.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/papers/loco_manip_161_survey_060_egohumanoid.md
  - ../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md
  - ../../sources/papers/humanoid_loco_manip_161_catalog.md
---

# EgoHumanoid

**EgoHumanoid** 收录于 [具身智能研究室 · 人形 Loco-Manip 161 篇长文](https://mp.weixin.qq.com/s/pACh9EhsISiyPGdiiR0C3A) **第 060/161** 篇，归类为 **03 视觉感知驱动的人形移动操作**。

## 一句话定义

EgoHumanoid 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹、遥操作/外骨骼数据编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型预测地形/场景表征。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Loco-Manip | Loco-Manipulation | 行走与操作动力学耦合的全身任务 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制层 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |

## 为什么重要

- EgoHumanoid 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹、遥操作/外骨骼数据编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型预测地形/场景表征。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。
- 人形 Loco-Manip 161 篇 **#060/161** · 视觉感知驱动的人形移动操作。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 060/161 |
| 分组 | 03 视觉感知驱动的人形移动操作 |
| 原文题目 | EgoHumanoid: Unlocking In-the-Wild Loco-Manipulation with Robot-Free Egocentric Demonstration |
| 机构 | The University of Hong Kong、Shanghai Innovation Institute、Beihang University、Robot teleoperation data collection is constrained to laboratory environment due to hardware and safety limitations, while in- |
| 发表日期 | 2026年6月4日 |
| 论文/项目 | https://github.com/NVlabs/GR00T-WholeBodyControl |

## 核心机制（归纳）

### 策展导读要点

EgoHumanoid 的实现路径是先把相机图像/多视角观测、人类视频/动捕轨迹、遥操作/外骨骼数据编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型预测地形/场景表征。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 评测与指标（索引级）

- 本条目为 161 篇策展索引级摘录，**未搬运原文量化 benchmark 与实机指标**；评测口径与具体数值以原文 PDF / 项目页为准。
- 评测原始出处：[原文 / 项目页](https://github.com/NVlabs/GR00T-WholeBodyControl)（见上方「核心信息」表「论文/项目」一行）。
- 横向评测对照请回到 [分类 hub](../overview/loco-manip-161-category-03-visuomotor.md) 与 [技术地图](../overview/humanoid-loco-manip-161-papers-technology-map.md)。

## 结论

**EgoHumanoid 押的是 in-the-wild 的第一视角、robot-free 示范：题名直指实验室遥操作受硬件与安全限制这一采集瓶颈，策展视角则把它读成多模态编码 + 模仿学习/VLA 的视觉驱动路线。**

- 机制主线：相机/多视角观测、人类视频/动捕轨迹、遥操作/外骨骼数据 → 多模态表征 → ACT/行为克隆与 VLA 动作模型，输出侧落在地形/场景表征，即在语义理解之外补机器人状态与动作头。
- 本页「论文/项目」一栏填的是 <https://github.com/NVlabs/GR00T-WholeBodyControl>，与题名及机构（港大 / 上海创智 / 北航）并不显然对应，机构栏还混入了摘要残句；引用前必须核对官方页面。
- 本条目为 **索引级坐标**（060/161 · 03 视觉感知驱动的人形移动操作，2026-06-04），未搬运量化 benchmark 与实机指标；in-the-wild 的可靠性最终取决于底层 WBC，单篇工作不自动解决。

## 常见误区

1. 161 篇策展条目提供 **地图坐标**；量化 benchmark 与实机指标以原文 PDF / 项目页为准。
2. Loco-manip 单篇工作不自动解决 **底层 WBC 鲁棒性**；须与运控/接触控制对照。

## 与其他页面的关系

- 技术地图：[humanoid-loco-manip-161-papers-technology-map.md](../overview/humanoid-loco-manip-161-papers-technology-map.md)
- 分类 hub：[loco-manip-161-category-03-visuomotor.md](../overview/loco-manip-161-category-03-visuomotor.md)
- 原始 source：[loco_manip_161_survey_060_egohumanoid.md](../../sources/papers/loco_manip_161_survey_060_egohumanoid.md)

## 参考来源

- [loco_manip_161_survey_060_egohumanoid.md](../../sources/papers/loco_manip_161_survey_060_egohumanoid.md) — 161 篇策展摘录
- [humanoid_loco_manip_161_catalog.md](../../sources/papers/humanoid_loco_manip_161_catalog.md)
- [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../../sources/blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)

## 推荐继续阅读

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
