---
type: entity
tags: [paper, underwater-robot, quadruped, attitude-control, hardware]
status: complete
updated: 2026-09-10
arxiv: "2609.09217"
code: https://github.com/ntnu-arl/uw-quadruped-cad
related:
  - ../methods/imitation-learning.md
  - ../methods/generative-world-models.md
  - ../tasks/manipulation.md
  - ../overview/vlm-manipulation-11-papers-technology-map.md
sources:
  - ../../sources/papers/underwater-quadruped-attitude-control_arxiv_2609_09217.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md
summary: "POM 防水电机壳+密封件+简化流体模型；SO(3) 误差闭环控制 roll/pitch/yaw；水箱真机验证。"
---

# 水下四足姿态控制（arXiv:2609.09217）

**水下四足姿态控制**（[Design and Attitude Control of an Underwater Quadruped Robot](https://arxiv.org/abs/2609.09217)）来自 [具身智能小站 11 篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md)。POM 防水电机壳+密封件+简化流体模型；SO(3) 误差闭环控制 roll/pitch/yaw；水箱真机验证。

## 一句话定义

**可复现硬件平台与姿态控制实验；控制代码截至入库日未确认开源。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IL | Imitation Learning | 从专家示范学习策略 |
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| WM | World Model | 预测未来观测或表征的动力学模型 |
| RL | Reinforcement Learning | 强化学习 |
| CEM | Cross-Entropy Method | 采样优化动作/轨迹的规划器 |
| DoF | Degrees of Freedom | 自由度 |

## 为什么重要

- 纳入本期 **VLM 控制 / 世界模型 / 灵巧操作 / 规划 / 评测** 主线之一。
- 开源状态：**部分开源**（步骤 2.5 核查，2026-09-10）。
- 与 [11 篇技术地图](../overview/vlm-manipulation-11-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.09217](https://arxiv.org/abs/2609.09217) |
| **项目页** | https://ntnu-arl.github.io/underwater-quadruped/ |
| **代码** | https://github.com/ntnu-arl/uw-quadruped-cad |
| **开源** | **部分开源** |
| **文内指标** | 可复现硬件平台与姿态控制实验；控制代码截至入库日未确认开源。 |


## 源码运行时序图

**不适用**（CAD/硬件描述已开源；控制栈以项目页与论文为主。）。


## 结论

**水下四足姿态控制 值得按「部分开源」边界阅读：先核对仓库是否可跑，再引用文内成功率数字。**

1. 索引来源为公众号导读，实验细节以 arXiv PDF 为准。
2. 开源结论：**部分开源** — CAD/硬件描述已开源；控制栈以项目页与论文为主。。
3. 选型时对照 [11 篇地图](../overview/vlm-manipulation-11-papers-technology-map.md) 中相邻节点，避免重复造页。

## 关联页面

- [VLM 与操作 11 篇技术地图](../overview/vlm-manipulation-11-papers-technology-map.md)
- [模仿学习 (Imitation Learning)](../methods/imitation-learning.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [underwater-quadruped-attitude-control_arxiv_2609_09217.md](../../sources/papers/underwater-quadruped-attitude-control_arxiv_2609_09217.md)
- [wechat 11篇盘点](../../sources/blogs/wechat_embodied_station_11_papers_vlm_manipulation_2026-09-10.md)
- [arXiv:2609.09217](https://arxiv.org/abs/2609.09217)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.09217)
- [项目页](https://ntnu-arl.github.io/underwater-quadruped/)
- [GitHub](https://github.com/ntnu-arl/uw-quadruped-cad)
