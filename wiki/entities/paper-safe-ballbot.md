---
type: entity
project_id: safe-ballbot
tags:
  - paper
  - mobile-manipulation
  - navigation
status: complete
updated: 2026-11-04
arxiv: "2611.03307"
project: https://safe-ballbot.github.io/
related:
  - ../tasks/vision-language-navigation.md
  - ../tasks/locomotion.md
  - ../tasks/manipulation.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
sources:
  - ../../sources/papers/safe_ballbot_arxiv_2611_03307.md
  - ../../sources/sites/safe-ballbot.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
summary: "Safe-Ballbot（arXiv:2611.03307）：用允许倾角推导规划加速度上界，连接上层MPC和下层平衡控制；结合控制屏障与预测误差膨胀避障，目前证据来自含轮接触和力矩饱和的仿真。"
---

# Safe-Ballbot（arXiv:2611.03307）

**Safe-Ballbot**（*Safe-Ballbot: Tilt-Consistent MPC with Control Barrier Functions for Ballbot Navigation*，[arXiv:2611.03307](https://arxiv.org/abs/2611.03307)，[项目页](https://safe-ballbot.github.io/)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「移动操作与导航」组。

## 一句话定义

**用允许倾角推导规划加速度上界，连接上层MPC和下层平衡控制；结合控制屏障与预测误差膨胀避障，目前证据来自含轮接触和力矩饱和的仿真。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Safe-Ballbot | Safe-Ballbot: Tilt-Consistent MPC with Control Barrier Functions for Ballbot Navigation | 本文方法 / 基准的简称 |

## 为什么重要

- 球形平衡机器人的避障不能忘了倾角。
- 开源结论：**部分开源**（步骤 2.5，2026-11-04）。公众号未给代码链接；项目页链接 asRoBallet/asRoBallet_mujoco 经 git ls-remote 可达，是否即论文完整代码未核对。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.03307](https://arxiv.org/abs/2611.03307) |
| **开源** | **部分开源** |
| **要点** | 用允许倾角推导规划加速度上界，连接上层MPC和下层平衡控制；结合控制屏障与预测误差膨胀避障，目前证据来自含轮接触和力矩饱和的仿真。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**部分开源** — 公众号未给代码链接；项目页链接 asRoBallet/asRoBallet_mujoco 经 git ls-remote 可达，是否即论文完整代码未核对。 待可运行代码公开后补图。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**球形平衡机器人的避障不能忘了倾角：用允许倾角推导规划加速度上界，连接上层MPC和下层平衡控制；结合控制屏障与预测误差膨胀避障，目前证据来自含轮接触和力矩饱和的仿真。**

1. 开源边界：**部分开源** — 公众号未给代码链接；项目页链接 asRoBallet/asRoBallet_mujoco 经 git ls-remote 可达，是否即论文完整代码未核对。
2. 核心机制：用允许倾角推导规划加速度上界，连接上层MPC和下层平衡控制；结合控制屏障与预测误差膨胀避障，目前证据来自含轮接触和力矩饱和的仿真。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [locomotion](../tasks/locomotion.md)
- [manipulation](../tasks/manipulation.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)

## 参考来源

- [safe_ballbot_arxiv_2611_03307.md](../../sources/papers/safe_ballbot_arxiv_2611_03307.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.03307](https://arxiv.org/abs/2611.03307)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.03307)
- [项目页](https://safe-ballbot.github.io/)
