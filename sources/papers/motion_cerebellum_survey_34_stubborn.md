# Stubborn 统一强化学习的人形动作跟踪与跌倒恢复

> 来源归档（ingest · 运动小脑 64 篇长文 第 34/64 + arXiv:2606.12814 深读）

- **标题：** Stubborn: A Streamlined and Unified Reinforcement Learning Framework for Robust Motion Tracking and Fall Recovery for Humanoids
- **类型：** paper
- **运动小脑分类：** D 全身跟踪基座（34/64）
- **机构：** 南方科技大学（ACT Lab）
- **arXiv：** <https://arxiv.org/abs/2606.12814>
- **项目页：** <https://aislab-sustech.github.io/Stubborn/>
- **入库日期：** 2026-06-18（策展索引）；2026-06-23（arXiv 深读增补）
- **一句话说明：** 用 **单一 RL 策略** 统一 **鲁棒运动跟踪** 与 **跌倒恢复**，核心为 **yaw-aligned 表征**、**Bernoulli 概率终止** 与 **跟踪误差驱动采样**，在 LAFAN1 全库与 G1 真机上验证。

## 核心摘录（策展，非全文）

- **在动作小脑地图中的位置：** D 全身跟踪基座，编号 **34/64**——「恢复：把跟踪和跌倒恢复放进统一 RL」。
- **统一框架动机：** 跟踪失败是自然进入倒地流形的入口；硬终止与分任务训练阻碍恢复探索；Stubborn 主张 **单策略、单阶段** 同时学跟踪与起身。
- **三大模块：**（1）yaw-aligned 漂移不变跟踪表征；（2）条件 Bernoulli **概率终止（PT）**；（3）**PT + 跟踪误差驱动自适应采样（AdpS）**。
- **量化亮点（LAFAN1 全库）：** MPBPE **48.85** mm、MPJPE **113.38**、MPJVE **624.03**；$\Delta$acc **17.09**（接近 BFM-Zero 且跟踪误差更低）。
- **恢复消融（5 m/s 扰动）：** 含 PT 时恢复成功率 **100%**（阈值 0.15 m / 0.25 m）；去掉 PT 为 77.5% / 85.0%。

## 对 wiki 的映射

- [paper-motion-cerebellum-stubborn](../../wiki/entities/paper-motion-cerebellum-stubborn.md)
- [motion-cerebellum-category-04-wbt-base](../../wiki/overview/motion-cerebellum-category-04-wbt-base.md)
- 深读 source：[stubborn_arxiv_2606_12814.md](./stubborn_arxiv_2606_12814.md)

## 参考来源（原始）

- 项目页：<https://aislab-sustech.github.io/Stubborn/>
- arXiv：<https://arxiv.org/abs/2606.12814>
- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
