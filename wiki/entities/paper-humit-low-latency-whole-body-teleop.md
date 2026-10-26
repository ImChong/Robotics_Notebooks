---
type: entity
tags:
  - paper
  - humanoid
status: complete
updated: 2026-10-26
arxiv: "2610.29903"
project_id: humit-low-latency-whole-body-teleop
project: https://xiaohu-art.github.io/Track/
related:
  - ../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/humit-low-latency-whole-body-teleop_arxiv_2610_29903.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md
summary: "HuMiT：基于紧凑参考跟踪的低延迟人形全身遥操作（arXiv:2610.29903）：控制器只接收根节点高度与速度，以及双腕、双踝四个关键点的当前 SE(3) 位姿；无需在线优化和未来参考，通信到策略输出延迟约 20 毫秒。"
---

# HuMiT：基于紧凑参考跟踪的低延迟人形全身遥操作

**HuMiT**（*HuMiT: Low-Latency Whole-Body Humanoid Teleoperation via Compact Reference Tracking*，[arXiv:2610.29903](https://arxiv.org/abs/2610.29903)）收录于 senlanke「具身运控lab」10-19–10-23 前沿论文动态（人形/四足）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**HuMiT：基于紧凑参考跟踪的低延迟人形全身遥操作——控制器只接收根节点高度与速度，以及双腕、双踝四个关键点的当前 SE(3) 位姿；无需在线优化和未来参考，通信到策略输出延迟约 20 毫秒。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 周报所述待解决的问题：密集全身参考、未来帧条件和在线动作重定向会显著增加遥操作延迟。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.29903](https://arxiv.org/abs/2610.29903) · [PDF](https://arxiv.org/pdf/2610.29903) |
| **作者（arXiv）** | Liu Cao, Botian Xu, Ruoqu Chen, Jiajun Liu, Mingzhi Pei, Zihao Wang, Zhongyu Li, Mengdi Xu |
| **机构（周报所述，未逐条核实）** | 清华大学、香港中文大学、中国人民大学 |
| **arXiv 评论栏** | 7 pages, 3 figures, 4 tables. Project page: this https URL |
| **开源** | arXiv 摘要页/评论给出项目页或外链（https://xiaohu-art.github.io/Track/）；未发现代码仓库链接（2026-10-26） |
| **周报所述创新点** | 控制器只接收根节点高度与速度，以及双腕、双踝四个关键点的当前 SE(3) 位姿；无需在线优化和未来参考，通信到策略输出延迟约 20 毫秒。 |

### arXiv 摘要节选（英文原文）

> Real-time whole-body humanoid teleoperation remains challenging due to the computational latency introduced by optimization-based retargeting pipelines and future-frame conditioning. Existing motion tracking policies typically require dense reference targets including full joint positions, body-part transformations, and even future frames, which tightly couples the controller to expensive upstream modules.

## 源码运行时序图

**不适用**：arXiv 摘要页/评论给出项目页或外链（https://xiaohu-art.github.io/Track/）；未发现代码仓库链接，无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [人形/四足周报技术地图](../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页/评论给出项目页或外链（https://xiaohu-art.github.io/Track/）；未发现代码仓库链接 |

## 结论

**HuMiT：基于紧凑参考跟踪的低延迟人形全身遥操作。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map](../overview/senlanke-weekly-2026-10-19-23-humanoid-quadruped-technology-map.md)
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [humit-low-latency-whole-body-teleop_arxiv_2610_29903.md](../../sources/papers/humit-low-latency-whole-body-teleop_arxiv_2610_29903.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
- [arXiv:2610.29903](https://arxiv.org/abs/2610.29903)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.29903)
