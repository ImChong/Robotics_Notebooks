---
type: entity
tags:
  - paper
  - humanoid
status: complete
updated: 2026-11-02
arxiv: "2610.36859"
project_id: activealign-egocentric-active-perception
related:
  - ../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/activealign-egocentric-active-perception_arxiv_2610_36859.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md
summary: "ActiveAlign（主动感知）：ActiveAlign：通过主动感知对齐第一视角机器人操作（arXiv:2610.36859）：利用主动挥手动作和可微渲染完成手眼标定，再通过主动触碰及触觉反馈细化物体位姿；在 Unitree G1 上完成按钮和滑块等毫米级操作。"
---

# ActiveAlign（主动感知）：ActiveAlign：通过主动感知对齐第一视角机器人操作

**ActiveAlign（主动感知）**（*ActiveAlign: Aligning Egocentric Manipulation via Active Perception*，[arXiv:2610.36859](https://arxiv.org/abs/2610.36859)）收录于 senlanke「具身运控lab」10-26–10-30 前沿论文动态（人形/四足）。本页依据**周报条目 + arXiv 摘要页**整理，未逐页核对正文 PDF；指标与结论以论文为准。

## 一句话定义

**ActiveAlign（主动感知）：ActiveAlign：通过主动感知对齐第一视角机器人操作——利用主动挥手动作和可微渲染完成手眼标定，再通过主动触碰及触觉反馈细化物体位姿；在 Unitree G1 上完成按钮和滑块等毫米级操作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 周报所述待解决的问题：被动采集的图像难以准确恢复相机外参和物体位置，导致真实场景与仿真场景错位，影响毫米级操作。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.36859](https://arxiv.org/abs/2610.36859) · [PDF](https://arxiv.org/pdf/2610.36859) |
| **作者（arXiv）** | Qiu, Ri-Zhao, Wen, Bowen, Jiang, Yifeng, Tessler, Chen, Heiden, Eric, Peng, Xue Bin, Wang, Xiaolong, Birchfield, Stan et al. |
| **机构（周报所述，未逐条核实）** | NVIDIA、加州大学圣迭戈分校 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02） |
| **周报所述创新点** | 利用主动挥手动作和可微渲染完成手眼标定，再通过主动触碰及触觉反馈细化物体位姿；在 Unitree G1 上完成按钮和滑块等毫米级操作。 |

### arXiv 摘要节选（英文原文）

> Real-to-Sim-to-Real (Real2Sim2Real) enables structured world representations for planning, interpretability, and safe execution. Most existing approaches rely on passive perception, which creates real2sim gaps due to imperfect observation.

## 源码运行时序图

**不适用**：arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布），无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [人形/四足周报技术地图](../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布） |

## 结论

**ActiveAlign（主动感知）：ActiveAlign：通过主动感知对齐第一视角机器人操作。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map](../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md)
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [activealign-egocentric-active-perception_arxiv_2610_36859.md](../../sources/papers/activealign-egocentric-active-perception_arxiv_2610_36859.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
- [arXiv:2610.36859](https://arxiv.org/abs/2610.36859)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.36859)
