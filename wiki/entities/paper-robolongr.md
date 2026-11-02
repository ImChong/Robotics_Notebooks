---
type: entity
project_id: robolongr
tags:
  - paper
  - vla
  - policy-learning
status: complete
updated: 2026-10-28
arxiv: "2610.33863"
project: https://chenzixuan99.github.io/RoboLongR-public/
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/reinforcement-learning.md
  - ../overview/embodied-station-2026-10-28-59-papers-technology-map.md
sources:
  - ../../sources/papers/robolongr_arxiv_2610_33863.md
  - ../../sources/sites/robolongr.md
  - ../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md
summary: "RoboLongR（arXiv:2610.33863）：用阶段乱序监督和循环记忆约束进度判断，避免只凭最终画面奖励漏步骤的执行。"
---

# RoboLongR（arXiv:2610.33863）

**RoboLongR**（*RoboLongR: Causally Consistent Rewards Teach Robots to Progressively Master Long-Horizon Manipulation*，[arXiv:2610.33863](https://arxiv.org/abs/2610.33863)，[项目页](https://chenzixuan99.github.io/RoboLongR-public/)）来自 [具身智能小站 2026-10-28 盘点](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)，属「VLA 与策略后训练」组。

## 一句话定义

**用阶段乱序监督和循环记忆约束进度判断，避免只凭最终画面奖励漏步骤的执行。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RoboLongR | RoboLongR: Causally Consistent Rewards Teach Robots to Progressively Master Long-Horizon Manipulation | 本文方法 / 基准的简称 |

## 为什么重要

- 长任务奖励需要记住前置步骤。结合行为克隆、在线强化学习与失败反馈纠偏，目标任务仍需示范和交互适配。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-10-28）。项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。
- 同期同组工作见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
  subgraph S1[1 因果一致奖励学习: RLBench 88 任务 + Open-X 示范]
    direction TB
    U[第1轮: 均匀分段, 目标 y=t/T 平方] --> AD[第2轮: 按预测进度曲线自适应分段<br/>Savitzky-Golay 平滑, 取曲率局部极大为边界]
    SH[stage-shuffle: 概率 0.5 将 S_i,S_i+1,S_i+2 重排为 S_i,S_i+2,S_i+1<br/>保留首个错位步之前的目标, 其后目标置 0] --> RM
    AG[增强: rewind / splice] --> RM
    RM[奖励模型 R_φ<br/>冻结 DINOv2 + MiniLM → 自回归 Transformer → 循环记忆 → MLP 进度头]
    U --> RM
    AD --> RM
  end
  S2[2 示范阶段标注<br/>按任务类别模板打粗阶段标签 z_t] --> S3
  S3[3 阶段条件初始化<br/>目标任务示范做行为克隆 BC<br/>策略输入含阶段嵌入] --> S4
  RM -->|进度分数作逐步奖励| S4[4 阶段调度在线 RL: PPO<br/>阶段从 Kc 逐级降到 1, 成功率达标才推进<br/>每回合从标准初始状态做完整任务]
  S4 --> FB{终端失败且<br/>最高奖励 > 0.65?}
  FB -->|同一任务连续 3 次| CORR[在线奖励纠偏<br/>对三条失败轨迹做一次梯度更新, 压低其奖励]
  CORR --> RM
```

图依据：论文 §III / Fig.2、§III-B 至 §III-E、式 (1)–(3)、Algorithm 1。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.33863](https://arxiv.org/abs/2610.33863) |
| **开源** | **项目页未列 GitHub** |
| **要点** | 用阶段乱序监督和循环记忆约束进度判断，避免只凭最终画面奖励漏步骤的执行。 |
| **文内指标** | 公众号转述：结合行为克隆、在线强化学习与失败反馈纠偏，目标任务仍需示范和交互适配。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号转述：结合行为克隆、在线强化学习与失败反馈纠偏，目标任务仍需示范和交互适配。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**长任务奖励需要记住前置步骤：结合行为克隆、在线强化学习与失败反馈纠偏，目标任务仍需示范和交互适配。**

1. 开源边界：**项目页未列 GitHub** — 项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。
2. 核心机制：用阶段乱序监督和循环记忆约束进度判断，避免只凭最终画面奖励漏步骤的执行。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [embodied-station-2026-10-28-59-papers-technology-map](../overview/embodied-station-2026-10-28-59-papers-technology-map.md)

## 参考来源

- [robolongr_arxiv_2610_33863.md](../../sources/papers/robolongr_arxiv_2610_33863.md)
- [wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md](../../sources/blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)
- [arXiv:2610.33863](https://arxiv.org/abs/2610.33863)
- [arXiv PDF](https://arxiv.org/pdf/2610.33863)：已通读方法部分（实验与附录以略读为准）：§I 引言、§III-A 问题形式化、§III-B 因果一致奖励学习、§III-C 示范阶段标注、§III-D 阶段条件初始化、§III-E 阶段调度在线 RL 与奖励纠偏、Fig.2、Algorithm 1；§IV 实验略读

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.33863)
- [项目页](https://chenzixuan99.github.io/RoboLongR-public/)
