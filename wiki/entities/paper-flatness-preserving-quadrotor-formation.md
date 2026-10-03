---
type: entity
tags: [paper, aerial, multi-robot, control, system-identification, iros-2026, upenn]
status: complete
updated: 2026-10-02
arxiv: "2607.12275"
related:
  - ../methods/reinforcement-learning.md
  - ../methods/trajectory-optimization.md
  - ../overview/iros-2026-awards-9-papers-technology-map.md
sources:
  - ../../sources/papers/flatness_preserving_quadrotor_formation_arxiv_2607_12275.md
  - ../../sources/blogs/wechat_iros_2026_awards_9_papers_2026-10-02.md
summary: "Flatness-Preserving Residual Learning（arXiv:2607.12275，IROS 2026 最佳学生论文）：保平坦性的编队残差动力学 + 反馈线性化，~28s 数据、5ms 环、跟踪误差 −31%，算力远低于 NMPC。"
---

# Flatness-Preserving Residual Learning（紧密四旋翼编队）

**Flatness-Preserving Residual Learning for Real-Time Tight Quadrotor Formation Flight**（[arXiv:2607.12275](https://arxiv.org/abs/2607.12275)，[视频](https://www.youtube.com/watch?v=uF26IkRFQMk)，**IROS 2026 最佳学生论文**）由 **宾夕法尼亚大学 GRASP Laboratory** 提出：在标称刚体模型上学习 **physics-informed 残差** 补偿下洗等气动干扰，并 **约束残差只依赖编队位姿/速度**，使多机系统仍 **微分平坦**，从而用 **反馈线性化 + 前馈** 实现 **5 ms** 级实时紧密编队。

## 一句话定义

**紧密编队的气动干扰可以学，但不能把平坦性学没——残差只挂在平坦坐标上，才能用轻量 FBL 控制器跑赢 NMPC 的算力账单。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FBL | Feedback Linearization | 反馈线性化控制 |
| NMPC | Nonlinear Model Predictive Control | 非线性模型预测控制 |
| RMSE | Root Mean Square Error | 均方根跟踪误差 |
| ROM | Reduced-Order Model | 下洗等降阶物理先验 |

## 为什么重要

- 纳入 [IROS 2026 九篇获奖盘点](../../sources/blogs/wechat_iros_2026_awards_9_papers_2026-10-02.md)：**传统控制 + 学习残差** 在 **IROS 主会最佳学生论文** 线获胜。
- 文内/摘要：**~28 s** 飞行数据、**5 ms** 控制周期、平均误差 **−31%** vs 标称；算力约为 NMPC **一个数量级** 更低。
- **开源结论（2026-10-02）：待发布** — 无官方 GitHub。

## 核心机制

| 项 | 内容 |
|----|------|
| **残差学习** | 在 ROM 下洗先验之上，小网络学剩余误差 |
| **平坦性约束** | 残差输入限制为编队 **位置/速度**，保联合系统平坦 |
| **控制器** | 预测扰动前馈 + 线性控制律；避免在线 NMPC 优化 |

## 源码运行时序图

**不适用**（截至入库日 arXiv/YouTube 可访问，**无官方可运行代码仓库**。）

## 实验与评测

- 硬件：紧密垂直间距编队、穿 **0.4 m** 高窗口等（视频演示）。
- **读法：** 公众号归纳；细节以 arXiv PDF 为准。

## 与其他工作对比

| 路线 | 扰动处理 | 在线算力 | 与本文差异 |
|------|----------|----------|------------|
| **标称刚体模型 + FBL** | 不建模下洗 | 低 | 本文报告平均跟踪误差 **−31%**（同一 FBL 框架加残差） |
| **[NMPC](../methods/model-predictive-control.md)** | 在线优化显式处理约束 | 高 | 本文算力约 **低一个数量级**；对比需对齐安全约束与编队间距口径 |
| **无约束残差学习** | 残差可依赖任意状态 | 视控制器而定 | 可能破坏微分平坦性，无法再用 FBL + 前馈；本文把残差输入限制在编队位姿/速度 |
| **端到端 [RL](../methods/reinforcement-learning.md) 编队** | 策略隐式吸收扰动 | 推理低、训练贵 | 本文只需 **~28 s** 飞行数据，结构可解释 |

## 结论

**保平坦性的残差动力学 = 可部署的紧密编队补偿** — 适合算力受限飞控栈，但依赖高质量短时飞行数据与 ROM 先验。

1. **待发布代码**：复现前只能对照 PDF/视频与自研仿真。
2. **31% 误差下降** 是相对 **标称基线**；与 NMPC 对比时注意 **同等安全约束** 与 **编队间距** 口径。
3. 与 **端到端 RL 编队** 对照：本路线强调 **可解释平坦坐标 + 毫秒级环**。
4. 部署前确认 **残差外推** 在更大编队规模/风场下是否仍安全。

## 关联页面

- [IROS 2026 九篇获奖地图](../overview/iros-2026-awards-9-papers-technology-map.md)
- [强化学习方法页](../methods/reinforcement-learning.md)
- [LT-Mem](./paper-lt-mem.md)（同盘点 · 最佳论文）

## 参考来源

- [IROS 2026 九篇获奖盘点（公众号）](../../sources/blogs/wechat_iros_2026_awards_9_papers_2026-10-02.md)
- [Flatness-Preserving Residual Learning（sources）](../../sources/papers/flatness_preserving_quadrotor_formation_arxiv_2607_12275.md)

## 推荐继续阅读

- arXiv:2607.12275 PDF：<https://arxiv.org/pdf/2607.12275>
